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
#include "llvm/Support/raw_ostream.h"

// Debug flags for MiniHLSL Interpreter
// - ENABLE_INTERPRETER_DEBUG: Shows detailed dynamic execution block tracking
// - ENABLE_WAVE_DEBUG: Shows wave operation synchronization details
// - ENABLE_BLOCK_DEBUG: Shows block creation, merging, and convergence

static constexpr bool ENABLE_INTERPRETER_DEBUG = true;  // Set to true to enable detailed execution tracing
static constexpr bool ENABLE_WAVE_DEBUG = true;        // Set to true to enable wave operation tracing  
static constexpr bool ENABLE_BLOCK_DEBUG = true;       // Set to true to enable block lifecycle tracing

#define INTERPRETER_DEBUG_LOG(msg) do { if (ENABLE_INTERPRETER_DEBUG) { llvm::errs() << msg; } } while(0)
#define WAVE_DEBUG_LOG(msg) do { if (ENABLE_WAVE_DEBUG) { llvm::errs() << msg; } } while(0)
#define BLOCK_DEBUG_LOG(msg) do { if (ENABLE_BLOCK_DEBUG) { llvm::errs() << msg; } } while(0)

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

// GlobalBuffer implementation
GlobalBuffer::GlobalBuffer(uint32_t size, const std::string& type) 
    : size_(size), bufferType_(type) {
  // Initialize buffer with default values (0)
  for (uint32_t i = 0; i < size_; ++i) {
    data_[i] = Value(0);
  }
}

Value GlobalBuffer::load(uint32_t index) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer access out of bounds: " << index << " >= " << size_);
    return Value(0);
  }
  return data_[index];
}

void GlobalBuffer::store(uint32_t index, const Value& value) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer access out of bounds: " << index << " >= " << size_);
    return;
  }
  data_[index] = value;
}

Value GlobalBuffer::atomicAdd(uint32_t index, const Value& value) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer atomic access out of bounds: " << index << " >= " << size_);
    return Value(0);
  }
  
  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) && std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::get<int32_t>(oldValue.data) + std::get<int32_t>(value.data));
  } else if (std::holds_alternative<float>(oldValue.data) && std::holds_alternative<float>(value.data)) {
    data_[index] = Value(std::get<float>(oldValue.data) + std::get<float>(value.data));
  } else {
    INTERPRETER_DEBUG_LOG("ERROR: Type mismatch in atomic add operation");
  }
  return oldValue;
}

Value GlobalBuffer::atomicSub(uint32_t index, const Value& value) {
  if (index >= size_) return Value(0);
  
  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) && std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::get<int32_t>(oldValue.data) - std::get<int32_t>(value.data));
  } else if (std::holds_alternative<float>(oldValue.data) && std::holds_alternative<float>(value.data)) {
    data_[index] = Value(std::get<float>(oldValue.data) - std::get<float>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicMin(uint32_t index, const Value& value) {
  if (index >= size_) return Value(0);
  
  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) && std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::min(std::get<int32_t>(oldValue.data), std::get<int32_t>(value.data)));
  } else if (std::holds_alternative<float>(oldValue.data) && std::holds_alternative<float>(value.data)) {
    data_[index] = Value(std::min(std::get<float>(oldValue.data), std::get<float>(value.data)));
  }
  return oldValue;
}

Value GlobalBuffer::atomicMax(uint32_t index, const Value& value) {
  if (index >= size_) return Value(0);
  
  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) && std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::max(std::get<int32_t>(oldValue.data), std::get<int32_t>(value.data)));
  } else if (std::holds_alternative<float>(oldValue.data) && std::holds_alternative<float>(value.data)) {
    data_[index] = Value(std::max(std::get<float>(oldValue.data), std::get<float>(value.data)));
  }
  return oldValue;
}

Value GlobalBuffer::atomicExchange(uint32_t index, const Value& value) {
  if (index >= size_) return Value(0);
  
  Value oldValue = data_[index];
  data_[index] = value;
  return oldValue;
}

Value GlobalBuffer::atomicCompareExchange(uint32_t index, const Value& compareValue, const Value& value) {
  if (index >= size_) return Value(0);
  
  Value oldValue = data_[index];
  if (oldValue.toString() == compareValue.toString()) {
    data_[index] = value;
  }
  return oldValue;
}

bool GlobalBuffer::hasConflictingAccess(uint32_t index, ThreadId tid1, ThreadId tid2) const {
  auto it = accessHistory_.find(index);
  if (it == accessHistory_.end()) return false;
  
  return it->second.count(tid1) > 0 && it->second.count(tid2) > 0;
}

std::map<uint32_t, Value> GlobalBuffer::getSnapshot() const {
  return data_;
}

void GlobalBuffer::clear() {
  data_.clear();
  accessHistory_.clear();
  for (uint32_t i = 0; i < size_; ++i) {
    data_[i] = Value(0);
  }
}

void GlobalBuffer::printContents() const {
  INTERPRETER_DEBUG_LOG("=== Global Buffer Contents (" << bufferType_ << ", size=" << size_ << ") ===");
  for (const auto& [index, value] : data_) {
    INTERPRETER_DEBUG_LOG("  [" << index << "] = " << value.toString());
  }
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
          createBlockIdentity(nullptr, BlockType::REGULAR, 0, {});

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
  switch (op_) {
  case Neg:
  case Minus: {
    Value val = expr_->evaluate(lane, wave, tg);
    return Value(-val.asFloat());
  }
  case Not:
  case LogicalNot: {
    Value val = expr_->evaluate(lane, wave, tg);
    return !val;
  }
  case Plus: {
    Value val = expr_->evaluate(lane, wave, tg);
    return val; // Unary plus does nothing
  }
  case PreIncrement: {
    // ++i: increment first, then return new value
    if (auto varExpr = dynamic_cast<const VariableExpr*>(expr_.get())) {
      std::string varName = varExpr->toString(); // Use toString() to get variable name
      Value& var = lane.variables[varName];
      var = Value(var.asInt() + 1);
      return var;
    }
    throw std::runtime_error("Pre-increment requires a variable");
  }
  case PostIncrement: {
    // i++: return old value, then increment
    if (auto varExpr = dynamic_cast<const VariableExpr*>(expr_.get())) {
      std::string varName = varExpr->toString(); // Use toString() to get variable name
      Value& var = lane.variables[varName];
      Value oldValue = var;
      var = Value(var.asInt() + 1);
      return oldValue;
    }
    throw std::runtime_error("Post-increment requires a variable");
  }
  case PreDecrement: {
    // --i: decrement first, then return new value
    if (auto varExpr = dynamic_cast<const VariableExpr*>(expr_.get())) {
      std::string varName = varExpr->toString(); // Use toString() to get variable name
      Value& var = lane.variables[varName];
      var = Value(var.asInt() - 1);
      return var;
    }
    throw std::runtime_error("Pre-decrement requires a variable");
  }
  case PostDecrement: {
    // i--: return old value, then decrement
    if (auto varExpr = dynamic_cast<const VariableExpr*>(expr_.get())) {
      std::string varName = varExpr->toString(); // Use toString() to get variable name
      Value& var = lane.variables[varName];
      Value oldValue = var;
      var = Value(var.asInt() - 1);
      return oldValue;
    }
    throw std::runtime_error("Post-decrement requires a variable");
  }
  }
  throw std::runtime_error("Unknown unary operator");
}

bool UnaryOpExpr::isDeterministic() const { return expr_->isDeterministic(); }

std::string UnaryOpExpr::toString() const {
  switch (op_) {
    case Neg:
    case Minus:
      return "-" + expr_->toString();
    case Not:
    case LogicalNot:
      return "!" + expr_->toString();
    case Plus:
      return "+" + expr_->toString();
    case PreIncrement:
      return "++" + expr_->toString();
    case PostIncrement:
      return expr_->toString() + "++";
    case PreDecrement:
      return "--" + expr_->toString();
    case PostDecrement:
      return expr_->toString() + "--";
  }
  return "?UnaryOp?";
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

  // Create block-scoped instruction identity using compound key
  uint32_t currentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  std::pair<const void*, uint32_t> instructionKey = {static_cast<const void*>(this), currentBlockId};
  
  std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " executing WaveActiveSum in block " 
            << currentBlockId << ", instruction key=(" << static_cast<const void*>(this) 
            << "," << currentBlockId << ")" << std::endl;

  // CRITICAL: If lane is resuming from wave operation, check for stored results first
  if (lane.isResumingFromWaveOp) {
    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " is resuming from wave operation, checking for stored result" << std::endl;
    
    auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
    if (syncPointIt != wave.activeSyncPoints.end()) {
      auto& syncPoint = syncPointIt->second;
      auto resultIt = syncPoint.pendingResults.find(lane.laneId);
      if (resultIt != syncPoint.pendingResults.end()) {
        // We have a result from collective execution - return it
        Value result = resultIt->second;
        std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " retrieving stored wave result: " 
                  << result.toString() << std::endl;
        
        // Remove this result to indicate it's been consumed
        syncPoint.pendingResults.erase(resultIt);
        
        // Clear the resuming flag - we successfully retrieved the result
        const_cast<LaneContext&>(lane).isResumingFromWaveOp = false;
        
        return result;
      }
    }
    
    // No stored result found - clear flag and continue with normal execution
    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " no stored result found for key (" << instructionKey.first << "," << instructionKey.second << "), continuing with normal execution" << std::endl;
    
    // Debug: show what results are actually available
    auto debugSyncPointIt = wave.activeSyncPoints.find(instructionKey);
    if (debugSyncPointIt != wave.activeSyncPoints.end()) {
      std::cout << "DEBUG: WAVE_OP: Available results for lanes: ";
      for (const auto& [availableLaneId, value] : debugSyncPointIt->second.pendingResults) {
        std::cout << availableLaneId << " ";
      }
      std::cout << std::endl;
    } else {
      std::cout << "DEBUG: WAVE_OP: No sync point found for this instruction key" << std::endl;
    }
    
    const_cast<LaneContext&>(lane).isResumingFromWaveOp = false;
  }

  // Check if there's already a computed result for this lane (normal path)
  auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
  if (syncPointIt != wave.activeSyncPoints.end()) {
    auto& syncPoint = syncPointIt->second;
    auto resultIt = syncPoint.pendingResults.find(lane.laneId);
    if (resultIt != syncPoint.pendingResults.end()) {
      // We have a result from collective execution - return it
      Value result = resultIt->second;
      INTERPRETER_DEBUG_LOG("Lane " << lane.laneId << " retrieving stored wave result: " 
                           << result.toString() << "\n");
      
      // Remove this result to indicate it's been consumed
      syncPoint.pendingResults.erase(resultIt);
      
      return result;
    }
  }

      // Mark this lane as arrived at this specific instruction  
    tg.markLaneArrivedAtWaveInstruction(wave.waveId, lane.laneId, static_cast<const void*>(this), "WaveActiveOp");
  // No stored result - check if we can execute or need to wait
  // In collective execution model, all lanes should wait for processWaveOperations to handle collective execution
  // Only skip waiting if we're resuming from a previous wave operation
  if (!lane.isResumingFromWaveOp) {
    
    // Store the compound key in the sync point for proper tracking
    auto& syncPoint = wave.activeSyncPoints[instructionKey];
    syncPoint.instruction = static_cast<const void*>(this);  // Store original instruction pointer

    // Get current block and mark lane as waiting
    tg.markLaneWaitingForWave(wave.waveId, lane.laneId, currentBlockId);
    
    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " cannot execute, starting to wait in block " 
              << currentBlockId << std::endl;
    
    // CRITICAL: Force refresh of block resolution status after marking lane as waiting
    // This is essential because the block needs to know all participants are now resolved
    auto blockIt = tg.executionBlocks.find(currentBlockId);
    if (blockIt != tg.executionBlocks.end()) {
      blockIt->second.setWaveAllUnknownResolved(wave.waveId, 
        blockIt->second.areAllUnknownLanesResolvedForWave(wave.waveId));
      std::cout << "DEBUG: WAVE_OP: Refreshed block " << currentBlockId << " resolution status" << std::endl;
    }
    
    // CRITICAL: 3-step logic for wave operation re-evaluation
    // Step 1: Check if this newly waiting lane completes the participant set
    // Step 2: Re-evaluate if wave operations can now execute with all participants waiting  
    // Step 3: If ready, mark sync point as complete so main loop will execute and wake lanes
    
    if (tg.canExecuteWaveInstruction(wave.waveId, lane.laneId, static_cast<const void*>(this))) {
      std::cout << "DEBUG: WAVE_OP: After lane " << lane.laneId << " started waiting, wave operation can now execute!" << std::endl;
      
      // Update the sync point to mark it as ready for execution
      syncPoint.allParticipantsArrived = (syncPoint.arrivedParticipants == syncPoint.expectedParticipants);
      
      // Re-check if all participants are known by checking block state
      auto blockIt = tg.executionBlocks.find(currentBlockId);
      if (blockIt != tg.executionBlocks.end()) {
        syncPoint.allParticipantsKnown = blockIt->second.isWaveAllUnknownResolved(wave.waveId);
      }
      
      // Mark as complete if both conditions are met
      syncPoint.isComplete = syncPoint.allParticipantsKnown && syncPoint.allParticipantsArrived;
      
      std::cout << "DEBUG: WAVE_OP: Updated sync point - allParticipantsKnown=" << syncPoint.allParticipantsKnown 
                << ", allParticipantsArrived=" << syncPoint.allParticipantsArrived 
                << ", isComplete=" << syncPoint.isComplete << std::endl;
    }
    
    // Throw a special exception to indicate we need to wait
    throw WaveOperationWaitException();
  }

  // This shouldn't happen in the new collective model, but keep as fallback
  std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId << " hit fallback path for instruction " 
            << instructionKey.first << " block " << instructionKey.second << std::endl;
  std::cout << "DEBUG: WAVE_OP: Lane state - isResumingFromWaveOp=" << lane.isResumingFromWaveOp << std::endl;
  std::cout << "DEBUG: WAVE_OP: Available sync points: " << wave.activeSyncPoints.size() << std::endl;
  for (const auto& [key, syncPoint] : wave.activeSyncPoints) {
    std::cout << "DEBUG: WAVE_OP: Sync point (" << key.first << "," << key.second << ") has " 
              << syncPoint.pendingResults.size() << " pending results" << std::endl;
  }
  throw std::runtime_error("Wave operation fallback path - should not reach here");
}

Value WaveActiveOp::computeWaveOperation(const std::vector<Value>& values) const {
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
  case AllTrue: {
    // Return true only if ALL participating lanes evaluate to true
    for (const auto &val : values) {
      if (!val.asBool()) {
        return Value(false);
      }
    }
    return Value(true);
  }
  case AnyTrue: {
    // Return true if ANY participating lane evaluates to true
    for (const auto &val : values) {
      if (val.asBool()) {
        return Value(true);
      }
    }
    return Value(false);
  }
  case AllEqual: {
    // Return true if all participating lanes have the same value
    if (values.empty()) {
      return Value(true);
    }
    const Value& firstValue = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      if (!(values[i] == firstValue)) {
        return Value(false);
      }
    }
    return Value(true);
  }
  case Ballot: {
    // Return a bitmask where bit N is set if lane N evaluates to true
    // Note: This is a simplified implementation that returns an integer
    // In real HLSL, this would return a 32-bit or 64-bit mask depending on wave size
    uint32_t ballot = 0;
    for (size_t i = 0; i < values.size() && i < 32; ++i) {
      if (values[i].asBool()) {
        ballot |= (1u << i);
      }
    }
    return Value(static_cast<int32_t>(ballot));
  }
  }
  throw std::runtime_error("Unknown wave operation");
}

std::string WaveActiveOp::toString() const {
  static const char *opNames[] = {"WaveActiveSum", "WaveActiveProduct",
                                  "WaveActiveMin", "WaveActiveMax",
                                  "WaveActiveBitAnd", "WaveActiveBitOr",
                                  "WaveActiveBitXor", "WaveActiveCountBits",
                                  "WaveActiveAllTrue", "WaveActiveAnyTrue",
                                  "WaveActiveAllEqual", "WaveActiveBallot"};
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

  try {
    Value initVal = init_ ? init_->evaluate(lane, wave, tg) : Value(0);
    lane.variables[name_] = initVal;
  } catch (const WaveOperationWaitException&) {
    // Lane is waiting for wave operation - re-throw so control flow statements can handle it
    std::cout << "DEBUG: VarDeclStmt - Lane " << lane.laneId << " caught WaveOperationWaitException, re-throwing" << std::endl;
    throw;  // Re-throw the exception
  }
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

  try {
    Value val = expr_->evaluate(lane, wave, tg);
    lane.variables[name_] = val;
  } catch (const WaveOperationWaitException&) {
    // Lane is waiting for wave operation - re-throw so control flow statements can handle it
    std::cout << "DEBUG: AssignStmt - Lane " << lane.laneId << " caught WaveOperationWaitException, re-throwing" << std::endl;
    throw;  // Re-throw the exception
  }
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

  // Find our entry in the execution stack (if any)
  int ourStackIndex = -1;
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    if (lane.executionStack[i].statement == static_cast<const void*>(this)) {
      ourStackIndex = i;
      break;
    }
  }
  
  bool isResuming = (ourStackIndex >= 0);

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(static_cast<const void*>(this), 
                                     LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " starting fresh execution (pushed to stack depth=" 
              << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
  } else {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " resuming execution (found at stack index=" 
              << ourStackIndex << ", current stack depth=" << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
  }

  // Don't hold reference to vector element - it can be invalidated during nested execution
  bool hasElse = !elseBlock_.empty();
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  bool setupComplete = (thenBlockId != 0 || elseBlockId != 0 || mergeBlockId != 0);

  try {
    while (lane.isActive) {
      // Use our entry, not back()
      auto& ourEntry = lane.executionStack[ourStackIndex];
      std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " in phase " << LaneContext::getPhaseString(ourEntry.phase) 
                << " (stack depth=" << lane.executionStack.size() << ", our index=" << ourStackIndex << ", this=" << this << ")" << std::endl;
      switch (ourEntry.phase) {
      
        case LaneContext::ControlFlowPhase::EvaluatingCondition: {
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " evaluating condition" << std::endl;
          
          // Only evaluate condition if not already evaluated (avoid re-evaluation on resume)
          if (!ourEntry.conditionEvaluated) {
            std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " evaluating condition for first time" << std::endl;
            // Evaluate condition (can throw WaveOperationWaitException)
            ourEntry.conditionResult = condition_->evaluate(lane, wave, tg).asBool();
            ourEntry.conditionEvaluated = true;
            std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " condition result=" << ourEntry.conditionResult << std::endl;
          } else {
            std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " using cached condition result=" << ourEntry.conditionResult << std::endl;
          }
          
          // Condition evaluated successfully - set up blocks
          std::set<uint32_t> divergentBlocks;
          tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                           parentBlockId, divergentBlocks);

          std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
          auto blockIds = tg.createIfBlocks(static_cast<const void *>(this), parentBlockId,
                                           currentMergeStack, hasElse, lane.executionPath);
          thenBlockId = std::get<0>(blockIds);
          elseBlockId = std::get<1>(blockIds);
          mergeBlockId = std::get<2>(blockIds);
          setupComplete = true;

          // Update blocks based on condition result
          if (ourEntry.conditionResult) {
            tg.moveThreadFromUnknownToParticipating(thenBlockId, wave.waveId, lane.laneId);
            if (hasElse) {
              tg.removeThreadFromUnknown(elseBlockId, wave.waveId, lane.laneId);
              tg.removeThreadFromNestedBlocks(elseBlockId, wave.waveId, lane.laneId);
            }
            // Don't remove from merge block yet - lane will reconverge there later
            
            ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingThenBlock;
            ourEntry.inThenBranch = true;
            ourEntry.blockId = thenBlockId;
          } else if (hasElse) {
            tg.moveThreadFromUnknownToParticipating(elseBlockId, wave.waveId, lane.laneId);
            tg.removeThreadFromUnknown(thenBlockId, wave.waveId, lane.laneId);
            tg.removeThreadFromNestedBlocks(thenBlockId, wave.waveId, lane.laneId);
            // Don't remove from merge block yet - lane will reconverge there later
            
            ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingElseBlock;
            ourEntry.inThenBranch = false;
            ourEntry.blockId = elseBlockId;
          } else {
            // No else block
            tg.removeThreadFromUnknown(thenBlockId, wave.waveId, lane.laneId);
            tg.removeThreadFromNestedBlocks(thenBlockId, wave.waveId, lane.laneId);
            tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
            
            ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
          }
          
          lane.executionStack[ourStackIndex].statementIndex = 0;
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " condition=" << ourEntry.conditionResult 
                    << ", moving to phase=" << LaneContext::getPhaseString(ourEntry.phase) << std::endl;
          break;
        }

        case LaneContext::ControlFlowPhase::ExecutingThenBlock: {
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " executing then block from statement " 
                    << ourEntry.statementIndex << std::endl;
          
          // Execute statements in then block from saved position
          for (size_t i = ourEntry.statementIndex; i < thenBlock_.size(); i++) {
            lane.executionStack[ourStackIndex].statementIndex = i;
            thenBlock_[i]->execute(lane, wave, tg);
            
            if (lane.hasReturned) {
              std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " popping stack due to return (depth " 
                        << lane.executionStack.size() << "->" << (lane.executionStack.size()-1) << ", this=" << this << ")" << std::endl;
              lane.executionStack.pop_back();
              tg.popMergePoint(wave.waveId, lane.laneId);
              return;
            }
            
            lane.executionStack[ourStackIndex].statementIndex = i + 1;
          }
          
          // Completed then block
          lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " completed then block, moving to reconvergence" << std::endl;
          break;
        }

        case LaneContext::ControlFlowPhase::ExecutingElseBlock: {
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " executing else block from statement " 
                    << ourEntry.statementIndex << std::endl;
          
          // Execute statements in else block from saved position
          for (size_t i = ourEntry.statementIndex; i < elseBlock_.size(); i++) {
            lane.executionStack[ourStackIndex].statementIndex = i;
            elseBlock_[i]->execute(lane, wave, tg);
            
            if (lane.hasReturned) {
              std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " popping stack due to return (depth " 
                        << lane.executionStack.size() << "->" << (lane.executionStack.size()-1) << ", this=" << this << ")" << std::endl;
              lane.executionStack.pop_back();
              tg.popMergePoint(wave.waveId, lane.laneId);
              return;
            }
            
            lane.executionStack[ourStackIndex].statementIndex = i + 1;
          }
          
          // Completed else block
          lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " completed else block, moving to reconvergence" << std::endl;
          break;
        }

        case LaneContext::ControlFlowPhase::Reconverging: {
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " performing reconvergence" << std::endl;
          
          // Use stored merge block ID - don't recreate blocks during reconvergence
          // mergeBlockId should already be set from initial setup
          
          // Clean up execution state
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " popping stack at reconvergence (depth " 
                    << lane.executionStack.size() << "->" << (lane.executionStack.size()-1) << ", this=" << this << ")" << std::endl;
          lane.executionStack.pop_back();
          
          // Reconverge at merge block
          tg.popMergePoint(wave.waveId, lane.laneId);
          tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
          
          // Move lane to merge block as participating (reconvergence)
          tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
          
          // Clean up then/else blocks - lane will never return to them
          if (setupComplete) {
            tg.removeThreadFromAllSets(thenBlockId, wave.waveId, lane.laneId);
            tg.removeThreadFromNestedBlocks(thenBlockId, wave.waveId, lane.laneId);
            
            if (hasElse && elseBlockId != 0) {
              tg.removeThreadFromAllSets(elseBlockId, wave.waveId, lane.laneId);
              tg.removeThreadFromNestedBlocks(elseBlockId, wave.waveId, lane.laneId);
            }
          }

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
          
          std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " reconvergence complete" << std::endl;
          return;
        }
      }
    }

  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - execution state is already saved
    // Note: ourEntry might be out of scope here, so we need to access via ourStackIndex
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " waiting for wave operation in phase " 
              << (int)lane.executionStack[ourStackIndex].phase << " at statement " << lane.executionStack[ourStackIndex].statementIndex << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    // Propagate break/continue to enclosing loop
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " popping stack due to ControlFlowException (depth " 
              << lane.executionStack.size() << "->" << (lane.executionStack.size()-1) << ", this=" << this << ")" << std::endl;
    lane.executionStack.pop_back();
    tg.popMergePoint(wave.waveId, lane.laneId);
    throw;
  }

  // Move lane to merge block as participating (reconvergence)
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
  
  // Clean up then/else blocks - lane will never return to them
  tg.removeThreadFromAllSets(thenBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(thenBlockId, wave.waveId, lane.laneId);
  
  if (hasElse && elseBlockId != 0) {
    tg.removeThreadFromAllSets(elseBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(elseBlockId, wave.waveId, lane.laneId);
  }

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

  // Find our entry in the execution stack (if any)
  int ourStackIndex = -1;
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    if (lane.executionStack[i].statement == static_cast<const void*>(this)) {
      ourStackIndex = i;
      break;
    }
  }
  
  bool isResuming = (ourStackIndex >= 0);
  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;
  uint32_t parentBlockId = 0;

  if (!isResuming) {
    // Starting fresh - push initial state for initialization
    lane.executionStack.emplace_back(static_cast<const void*>(this), 
                                     LaneContext::ControlFlowPhase::EvaluatingInit);
    ourStackIndex = lane.executionStack.size() - 1;
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " starting fresh execution (pushed to stack depth=" 
              << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
              
    // Get current block before entering loop
    parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

    // Get current merge stack for block creation
    std::vector<MergeStackEntry> currentMergeStack =
        tg.getCurrentMergeStack(wave.waveId, lane.laneId);

    // Create loop blocks (header, merge) - pass current execution path
    auto [hBlockId, mBlockId] =
        tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                            currentMergeStack, lane.executionPath);
    headerBlockId = hBlockId;
    mergeBlockId = mBlockId;
    
    // Save block IDs in our stack entry
    lane.executionStack[ourStackIndex].loopHeaderBlockId = headerBlockId;
    lane.executionStack[ourStackIndex].loopMergeBlockId = mergeBlockId;

    // Push merge point for loop divergence
    std::set<uint32_t> divergentBlocks = {headerBlockId};
    tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                      parentBlockId, divergentBlocks);
  } else {
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " resuming execution (found at stack index=" 
              << ourStackIndex << ", current stack depth=" << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
    
    // Restore saved block IDs
    headerBlockId = lane.executionStack[ourStackIndex].loopHeaderBlockId;
    mergeBlockId = lane.executionStack[ourStackIndex].loopMergeBlockId;
  }

  // Execute loop with state machine
  try {
    while (lane.isActive) {
      auto& ourEntry = lane.executionStack[ourStackIndex];
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " in phase " << LaneContext::getPhaseString(ourEntry.phase) 
                << " (stack depth=" << lane.executionStack.size() << ", our index=" << ourStackIndex << ", this=" << this << ")" << std::endl;
                
      switch (ourEntry.phase) {
        case LaneContext::ControlFlowPhase::EvaluatingInit: {
          std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating init" << std::endl;
          
          // Initialize loop variable
          lane.variables[loopVar_] = init_->evaluate(lane, wave, tg);
          
          // Move to loop header block
          tg.assignLaneToBlock(wave.waveId, lane.laneId, headerBlockId);
          tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
          
          // Move to condition evaluation phase
          lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
          ourEntry.loopIteration = 0;
          break;
        }
        
        case LaneContext::ControlFlowPhase::EvaluatingCondition: {
          std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating condition for iteration " 
                    << ourEntry.loopIteration << std::endl;
          
          // Check loop condition
          bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
          if (!shouldContinue) {
            // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
            tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);      // Remove from header
            tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId); // Remove from iteration blocks
            
            // Move to reconverging phase
            ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
            break;
          }
          
          // Condition passed, move to body execution
          ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
          lane.executionStack[ourStackIndex].statementIndex = 0;
          break;
        }
        
        case LaneContext::ControlFlowPhase::ExecutingBody: {
          std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing body for iteration " 
                    << ourEntry.loopIteration << " from statement " << ourEntry.statementIndex << std::endl;
          
          // Execute statements in body from saved position
          for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
            lane.executionStack[ourStackIndex].statementIndex = i;
            body_[i]->execute(lane, wave, tg);
            
            if (lane.hasReturned) {
              lane.executionStack.pop_back();
              tg.popMergePoint(wave.waveId, lane.laneId);
              return;
            }
            
            // Re-fetch reference in case vector reallocated during statement execution
            lane.executionStack[ourStackIndex].statementIndex = i + 1;
          }
          
          // Body completed, move to increment phase
          lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;
          break;
        }
        
        case LaneContext::ControlFlowPhase::EvaluatingIncrement: {
          std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating increment for iteration " 
                    << ourEntry.loopIteration << std::endl;
          
          // Increment loop variable
          lane.variables[loopVar_] = increment_->evaluate(lane, wave, tg);
          
          // Move to next iteration
          lane.executionStack[ourStackIndex].loopIteration++;
          lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
          break;
        }
        
        case LaneContext::ControlFlowPhase::Reconverging: {
          std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " exiting loop after " 
                    << ourEntry.loopIteration << " iterations" << std::endl;
          
          // Clean up execution state
          lane.executionStack.pop_back();
          
          // Pop merge point and move to merge block
          tg.popMergePoint(wave.waveId, lane.laneId);
          tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
          tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
          return;
        }
        
        
        default:
          std::cout << "ERROR: ForStmt - Unexpected phase " << static_cast<int>(ourEntry.phase) << std::endl;
          return;
      }
    }
  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - execution state is already saved
    auto& ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " waiting for wave operation in phase " 
              << static_cast<int>(ourEntry.phase) << " at statement " << ourEntry.statementIndex 
              << ", iteration " << ourEntry.loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      // Break - exit loop
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " breaking from loop" << std::endl;
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      
      // Clean up - remove from blocks this lane will never reach
      tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);
      
      tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
      tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      // Continue - go to increment phase
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " continuing loop" << std::endl;
      auto& ourEntry = lane.executionStack[ourStackIndex];
      lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;
    }
  }
  
  // If we reach here, continue the loop by recursively calling execute
  // This handles the state machine transitions
  try {
    execute(lane, wave, tg);
  } catch (const WaveOperationWaitException& e) {
    // Propagate wave operation wait to parent
    throw;
  }
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
            const auto& instructionKey = waitingIt->second;
            canProceed = tg.canExecuteWaveInstruction(waveId, laneId, instructionKey.first);
          }

          if (canProceed) {
            wave.lanes[laneId]->state = ThreadState::Ready;
            wave.lanes[laneId]->isResumingFromWaveOp = true;  // Set resuming flag
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
  std::vector<std::pair<const void*, uint32_t>>
      completedInstructions; // Track instructions that become complete

  for (auto &[instructionKey, syncPoint] : wave.activeSyncPoints) {
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
      completedInstructions.push_back(instructionKey);
    }
  }

  // Wake up lanes waiting at newly completed sync points
  for (const auto& instructionKey : completedInstructions) {
    auto &syncPoint = wave.activeSyncPoints[instructionKey];
    for (LaneId waitingLaneId : syncPoint.arrivedParticipants) {
      if (waitingLaneId < wave.lanes.size() && wave.lanes[waitingLaneId] &&
          wave.lanes[waitingLaneId]->state == ThreadState::WaitingForWave) {
        wave.lanes[waitingLaneId]->state = ThreadState::Ready;
        wave.lanes[waitingLaneId]->isResumingFromWaveOp = true;  // Set resuming flag
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
    // Check if this thread was supposed to participate in the barrier
    if (barrier.participatingThreads.count(returningThreadId) > 0) {
      // DEADLOCK ERROR: Thread returned early without hitting barrier
      std::stringstream errorMsg;
      errorMsg << "Deadlock: Thread " << returningThreadId 
               << " returned early without reaching barrier " << barrierId
               << ". All threads must reach the same barrier.";
      
      INTERPRETER_DEBUG_LOG("ERROR: " << errorMsg.str());
      
      // Mark ALL remaining threads as error state - this is undefined behavior
      for (auto& wave : tg.waves) {
        for (auto& lane : wave->lanes) {
          if (lane->state != ThreadState::Completed && lane->state != ThreadState::Error) {
            lane->state = ThreadState::Error;
            lane->errorMessage = errorMsg.str();
          }
        }
      }
      
      // Clear all barriers since the program is now in error state
      tg.activeBarriers.clear();
      return;
    }
    
    // If thread wasn't participating in this barrier, just remove it from arrived list
    barrier.arrivedThreads.erase(returningThreadId);
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
  if (!lane.isActive)
    return;
    
  INTERPRETER_DEBUG_LOG("Lane " << lane.laneId << " in wave " << wave.waveId << " hitting barrier");
  
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  
  // Create a new barrier or join existing one for collective execution
  uint32_t barrierId = tg.nextBarrierId;
  
  // Check if there's already an active barrier for this threadgroup
  bool foundActiveBarrier = false;
  for (auto& [id, barrier] : tg.activeBarriers) {
    // Join the first active barrier (since all threads must reach the same barrier)
    barrierId = id;
    foundActiveBarrier = true;
    break;
  }
  
  if (!foundActiveBarrier) {
    // Create new barrier for collective execution
    ThreadgroupBarrierState newBarrier;
    newBarrier.barrierId = barrierId;
    
    // ALL threads in threadgroup must participate in barriers (even if some returned early)
    // This is different from wave operations which only include active lanes
    // Barriers require ALL threads that started execution to reach the barrier
    for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
      for (size_t laneId = 0; laneId < tg.waves[waveId]->lanes.size(); ++laneId) {
        // Include ALL threads, regardless of current state
        // If a thread returned early, it will trigger deadlock detection
        ThreadId participantTid = tg.getGlobalThreadId(waveId, laneId);
        newBarrier.participatingThreads.insert(participantTid);
      }
    }
    
    tg.activeBarriers[barrierId] = newBarrier;
    tg.nextBarrierId++;
    
    INTERPRETER_DEBUG_LOG("Created collective barrier " << barrierId << " expecting " 
                         << newBarrier.participatingThreads.size() << " threads");
  }
  
  // Add this thread to the barrier for collective execution
  auto& barrier = tg.activeBarriers[barrierId];
  barrier.arrivedThreads.insert(tid);
  
  // Set thread state to waiting for collective barrier execution
  lane.state = ThreadState::WaitingAtBarrier;
  lane.waitingBarrierId = barrierId;
  
  INTERPRETER_DEBUG_LOG("Thread " << tid << " waiting for collective barrier " << barrierId 
                       << " (" << barrier.arrivedThreads.size() << "/" 
                       << barrier.participatingThreads.size() << " arrived)");
  
  // If all threads have arrived, the barrier will execute collectively in processBarriers()
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

Value BufferAccessExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
  // Evaluate the index expression
  Value indexValue = indexExpr_->evaluate(lane, wave, tg);
  uint32_t index = indexValue.asInt();
  
  // Look up the global buffer
  auto bufferIt = tg.globalBuffers.find(bufferName_);
  if (bufferIt == tg.globalBuffers.end()) {
    throw std::runtime_error("Global buffer not found: " + bufferName_);
  }
  
  return bufferIt->second->load(index);
}

std::string BufferAccessExpr::toString() const {
  return bufferName_ + "[" + indexExpr_->toString() + "]";
}

// MiniHLSLInterpreter implementation with cooperative scheduling
ExecutionResult
MiniHLSLInterpreter::executeWithOrdering(const Program &program,
                                         const ThreadOrdering &ordering,
                                         uint32_t waveSize) {
  ExecutionResult result;

  // Create threadgroup context
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
        bool hasWaitingThreads = false;
        for (const auto &wave : tgContext.waves) {
          for (const auto &lane : wave->lanes) {
            if (lane->state == ThreadState::WaitingForWave ||
                lane->state == ThreadState::WaitingAtBarrier) {
              hasWaitingThreads = true;
            }
            if (lane->state != ThreadState::Completed &&
                lane->state != ThreadState::Error) {
              allCompleted = false;
            }
          }
        }

        if (allCompleted) {
          break; // All threads finished
        }

        // If we have waiting threads but no ready threads, 
        // continue to let processWaveOperations wake them up
        if (hasWaitingThreads) {
          continue; // Let synchronization complete
        }

        // Check for true deadlock (no waiting, no ready, not completed)
        result.errorMessage =
            "Deadlock detected: no threads ready or waiting";
        break;
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

  // Print Dynamic Block Execution Graph (DBEG) for debugging merge blocks
  tgContext.printDynamicExecutionGraph(true);

  // Print final variable values for all lanes
  tgContext.printFinalVariableValues();

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
    
    // Only increment if the lane is not waiting
    if (lane.state == ThreadState::Ready) {
      lane.currentStatement++;
    }

    if (lane.hasReturned) {
      lane.state = ThreadState::Completed;
    }
  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - do nothing, statement will be retried
    // Lane state should already be set to WaitingForWave
    std::cout << "DEBUG: WAVE_WAIT: Lane " << (tid % 32) << " caught WaveOperationWaitException, state=" 
              << (int)lane.state << std::endl;
  } catch (const std::exception &e) {
    lane.state = ThreadState::Error;
    lane.errorMessage = e.what();
  }

  return true;
}

void MiniHLSLInterpreter::processWaveOperations(ThreadgroupContext &tgContext) {
  // Process wave operations for each wave
  for (size_t waveId = 0; waveId < tgContext.waves.size(); ++waveId) {
    auto& wave = *tgContext.waves[waveId];
    
    // Check active sync points for this wave
    std::vector<std::pair<const void*, uint32_t>> completedSyncPoints;
    
    // Find sync points that are ready to execute
    for (auto& [instructionKey, syncPoint] : wave.activeSyncPoints) {
      // Before checking if complete, update participants to include all current lanes in the block
      if (syncPoint.isComplete && syncPoint.pendingResults.empty()) {
        // Update arrivedParticipants to include all lanes currently in this block
        uint32_t blockId = instructionKey.second;
        auto blockIt = tgContext.executionBlocks.find(blockId);
        if (blockIt != tgContext.executionBlocks.end()) {
          const auto& blockLanes = blockIt->second.getParticipatingLanes();
          auto waveIt = blockLanes.find(waveId);
          if (waveIt != blockLanes.end()) {
            // Update participants to include all current lanes in block
            for (LaneId laneId : waveIt->second) {
              syncPoint.arrivedParticipants.insert(laneId);
              syncPoint.expectedParticipants.insert(laneId);
            }
            std::cout << "DEBUG: WAVE_OP: Updated participants before execution to include all current block lanes: ";
            for (LaneId laneId : syncPoint.arrivedParticipants) {
              std::cout << laneId << " ";
            }
            std::cout << std::endl;
          }
        }
        INTERPRETER_DEBUG_LOG("Executing collective wave operation for wave " << waveId 
                             << " at instruction " << instructionKey.first << " block " << instructionKey.second << "\n");
        
        // Execute the wave operation collectively
        executeCollectiveWaveOperation(tgContext, waveId, instructionKey, syncPoint);
        
        // Don't release yet - let lanes retrieve results first
      } else if (syncPoint.isComplete && !syncPoint.pendingResults.empty()) {
        // Results are available, wake up lanes so they can retrieve them
        INTERPRETER_DEBUG_LOG("Waking up lanes to retrieve wave operation results for wave " << waveId << "\n");
        for (LaneId laneId : syncPoint.arrivedParticipants) {
          if (laneId < wave.lanes.size() && 
              wave.lanes[laneId]->state == ThreadState::WaitingForWave) {
            INTERPRETER_DEBUG_LOG("  Waking up lane " << laneId 
                                 << " from WaitingForWave to Ready\n");
            wave.lanes[laneId]->state = ThreadState::Ready;
            wave.lanes[laneId]->isResumingFromWaveOp = true;  // Set resuming flag
            // Also remove from the waiting instruction map
            wave.laneWaitingAtInstruction.erase(laneId);
          }
        }
        
        // Mark for release after lanes retrieve results
        completedSyncPoints.push_back(instructionKey);
      }
    }
    
    // Release completed sync points after lanes have retrieved their results
    for (const auto& instructionKey : completedSyncPoints) {
      // Only release if all results have been consumed
      auto& syncPoint = wave.activeSyncPoints[instructionKey];
      
      if (syncPoint.pendingResults.empty()) {
        INTERPRETER_DEBUG_LOG("All results consumed, releasing sync point for wave " << waveId 
                             << " at instruction " << instructionKey.first << " block " << instructionKey.second << "\n");
        tgContext.releaseWaveSyncPoint(waveId, instructionKey);
      }
    }
  }
}

void MiniHLSLInterpreter::executeCollectiveWaveOperation(ThreadgroupContext &tgContext, 
                                                       WaveId waveId, 
                                                       const std::pair<const void*, uint32_t>& instructionKey, 
                                                       WaveOperationSyncPoint &syncPoint) {
  // Get the actual WaveActiveOp object from the original instruction pointer stored in syncPoint
  const WaveActiveOp* waveOp = static_cast<const WaveActiveOp*>(syncPoint.instruction);
  
  // Collect values from all participating lanes
  std::vector<Value> values;
  auto& wave = *tgContext.waves[waveId];
  
  for (LaneId laneId : syncPoint.arrivedParticipants) {
    if (laneId < wave.lanes.size()) {
      // Evaluate the expression for this lane
      Value value = waveOp->getExpression()->evaluate(*wave.lanes[laneId], wave, tgContext);
      values.push_back(value);
    }
  }
  
  // Execute the wave operation on the collected values
  Value result = waveOp->computeWaveOperation(values);
  
  // Store the result for all participating lanes
  std::cout << "DEBUG: WAVE_OP: Storing collective result for lanes: ";
  for (LaneId laneId : syncPoint.arrivedParticipants) {
    std::cout << laneId << " ";
    syncPoint.pendingResults[laneId] = result;
  }
  std::cout << std::endl;
  
  INTERPRETER_DEBUG_LOG("Collective wave operation result: " << result.toString() << "\n");
}

void MiniHLSLInterpreter::executeCollectiveBarrier(ThreadgroupContext &tgContext, 
                                                   uint32_t barrierId, 
                                                   const ThreadgroupBarrierState &barrier) {
  INTERPRETER_DEBUG_LOG("Executing collective barrier " << barrierId 
                       << " with " << barrier.arrivedThreads.size() << " threads");
  
  // Collective barrier execution:
  // 1. Memory fence - ensure all prior memory operations are visible
  // 2. Synchronization point - all threads wait here
  // 3. Barrier semantics - GroupMemoryBarrierWithGroupSync()
  
  // In a real GPU, this would:
  // - Flush caches
  // - Ensure memory consistency across threadgroup
  // - Synchronize all threads at this execution point
  
  // For our interpreter, the collective barrier execution is conceptual
  // The important part is that ALL threads execute this together
  INTERPRETER_DEBUG_LOG("Collective barrier " << barrierId << " memory fence and sync complete");
}

void MiniHLSLInterpreter::processBarriers(ThreadgroupContext &tgContext) {
  std::vector<uint32_t> completedBarriers;
  
  // Check each active barrier to see if it's complete or deadlocked
  for (auto& [barrierId, barrier] : tgContext.activeBarriers) {
    if (barrier.arrivedThreads.size() == barrier.participatingThreads.size()) {
      // All threads have arrived - execute barrier collectively
      INTERPRETER_DEBUG_LOG("Collective barrier " << barrierId << " executing! All " 
                           << barrier.arrivedThreads.size() << " threads synchronized");
      
      // Execute barrier collectively (memory fence, synchronization point)
      executeCollectiveBarrier(tgContext, barrierId, barrier);
      
      // Release all waiting threads simultaneously
      for (ThreadId tid : barrier.arrivedThreads) {
        auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
        if (waveId < tgContext.waves.size() && laneId < tgContext.waves[waveId]->lanes.size()) {
          auto& lane = tgContext.waves[waveId]->lanes[laneId];
          if (lane->state == ThreadState::WaitingAtBarrier && lane->waitingBarrierId == barrierId) {
            lane->state = ThreadState::Ready;
            lane->waitingBarrierId = 0;
            // Advance to next statement after barrier completion
            lane->currentStatement++;
            INTERPRETER_DEBUG_LOG("Released thread " << tid << " from barrier " << barrierId 
                                 << " and advanced to statement " << lane->currentStatement);
          }
        }
      }
      
      completedBarriers.push_back(barrierId);
    } else {
      // Check for deadlock - if some threads are completed/error but not all expected threads arrived
      std::set<ThreadId> stillExecutingThreads;
      for (ThreadId tid : barrier.participatingThreads) {
        auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
        if (waveId < tgContext.waves.size() && laneId < tgContext.waves[waveId]->lanes.size()) {
          auto& lane = tgContext.waves[waveId]->lanes[laneId];
          if (lane->state != ThreadState::Completed && lane->state != ThreadState::Error) {
            stillExecutingThreads.insert(tid);
          }
        }
      }
      
      // If some threads that should participate are no longer executing, we have a deadlock
      if (stillExecutingThreads.size() < barrier.participatingThreads.size() && 
          barrier.arrivedThreads.size() < stillExecutingThreads.size()) {
        // Deadlock detected - some threads completed without hitting barrier
        std::stringstream errorMsg;
        errorMsg << "Barrier deadlock detected! Barrier " << barrierId 
                 << " expected " << barrier.participatingThreads.size() << " threads, "
                 << "but only " << barrier.arrivedThreads.size() << " arrived. "
                 << "Some threads completed execution without reaching the barrier.";
        
        INTERPRETER_DEBUG_LOG("ERROR: " << errorMsg.str());
        
        // Mark all remaining threads as error state
        for (ThreadId tid : stillExecutingThreads) {
          auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
          if (waveId < tgContext.waves.size() && laneId < tgContext.waves[waveId]->lanes.size()) {
            auto& lane = tgContext.waves[waveId]->lanes[laneId];
            lane->state = ThreadState::Error;
            lane->errorMessage = errorMsg.str();
          }
        }
        
        completedBarriers.push_back(barrierId);
      }
    }
  }
  
  // Remove completed barriers
  for (uint32_t barrierId : completedBarriers) {
    tgContext.activeBarriers.erase(barrierId);
  }
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
                                             uint32_t numOrderings,
                                             uint32_t waveSize) {
  VerificationResult verification;

  // Generate test orderings
  verification.orderings =
      generateTestOrderings(program.getTotalThreads(), numOrderings);

  // Execute with each ordering
  for (const auto &ordering : verification.orderings) {
    verification.results.push_back(executeWithOrdering(program, ordering, waveSize));
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
                                             const ThreadOrdering &ordering,
                                             uint32_t waveSize) {
  return executeWithOrdering(program, ordering, waveSize);
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
  // Check CompoundAssignOperator first since it inherits from BinaryOperator
  if (auto compoundOp = clang::dyn_cast<clang::CompoundAssignOperator>(stmt)) {
    return convertCompoundAssignOperator(compoundOp, context);
  } else if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(stmt)) {
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
  } else if (auto returnStmt = clang::dyn_cast<clang::ReturnStmt>(stmt)) {
    return convertReturnStatement(returnStmt, context);
  } else if (auto unaryOp = clang::dyn_cast<clang::UnaryOperator>(stmt)) {
    // Handle unary operators like i++ as expression statements
    auto expr = convertExpression(unaryOp, context);
    if (expr) {
      return std::make_unique<ExprStmt>(std::move(expr));
    }
    return nullptr;
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
MiniHLSLInterpreter::convertCompoundAssignOperator(const clang::CompoundAssignOperator *compoundOp,
                                                   clang::ASTContext &context) {
  std::cout << "Converting compound assignment operator" << std::endl;
  
  // Get the target variable name
  std::string targetVar = "unknown";
  if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(compoundOp->getLHS())) {
    targetVar = declRef->getDecl()->getName().str();
  }
  
  // Convert compound assignment to regular assignment
  // For example: result += 1 becomes result = result + 1
  auto lhs = convertExpression(compoundOp->getLHS(), context);
  auto rhs = convertExpression(compoundOp->getRHS(), context);
  
  if (!lhs || !rhs) {
    std::cout << "Failed to convert compound assignment operands" << std::endl;
    return nullptr;
  }
  
  // Determine the binary operation based on compound assignment type
  BinaryOpExpr::OpType opType;
  switch (compoundOp->getOpcode()) {
    case clang::BO_AddAssign:
      opType = BinaryOpExpr::Add;
      break;
    case clang::BO_SubAssign:
      opType = BinaryOpExpr::Sub;
      break;
    case clang::BO_MulAssign:
      opType = BinaryOpExpr::Mul;
      break;
    case clang::BO_DivAssign:
      opType = BinaryOpExpr::Div;
      break;
    default:
      std::cout << "Unsupported compound assignment operator" << std::endl;
      return nullptr;
  }
  
  // Create the binary expression: lhs op rhs
  auto binaryExpr = std::make_unique<BinaryOpExpr>(std::move(lhs), std::move(rhs), opType);
  
  return makeAssign(targetVar, std::move(binaryExpr));
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
    } else if (funcName == "WaveActiveBitAnd" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::And));
      }
    } else if (funcName == "WaveActiveBitOr" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or));
      }
    } else if (funcName == "WaveActiveBitXor" && callExpr->getNumArgs() == 1) {
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
    } else if (funcName == "WaveActiveAllTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::AllTrue));
      }
    } else if (funcName == "WaveActiveAnyTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::AnyTrue));
      }
    } else if (funcName == "WaveActiveAllEqual" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::AllEqual));
      }
    } else if (funcName == "WaveActiveBallot" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::Ballot));
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

      // Create a variable declaration with or without initializer
      if (varDecl->hasInit()) {
        auto initExpr = convertExpression(varDecl->getInit(), context);
        if (initExpr) {
          return makeVarDecl(varName, std::move(initExpr));
        }
      } else {
        // Variable declaration without initializer
        return makeVarDecl(varName, nullptr);
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
        std::cout << "DEBUG: Switch parsing - processing AST node: " 
                  << stmt->getStmtClassName() << std::endl;
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
            std::cout << "DEBUG: Switch parsing - LHS type: " << lhs->getStmtClassName() << std::endl;
            // Handle implicit casts
            auto unwrapped = lhs->IgnoreImpCasts();
            if (auto intLit = clang::dyn_cast<clang::IntegerLiteral>(unwrapped)) {
              currentCaseValue = intLit->getValue().getSExtValue();
              std::cout << "DEBUG: Switch parsing - found case " << currentCaseValue.value() << std::endl;
            } else {
              std::cout << "DEBUG: Switch parsing - unwrapped LHS type: " << unwrapped->getStmtClassName() << std::endl;
            }
          } else {
            std::cout << "DEBUG: Switch parsing - no LHS found" << std::endl;
          }

          // Convert case body
          if (auto substmt = caseStmt->getSubStmt()) {
            if (auto converted = convertStatement(substmt, context)) {
              currentCase.push_back(std::move(converted));
            }
          }
        } else if (auto defaultStmt = clang::dyn_cast<clang::DefaultStmt>(stmt)) {
          std::cout << "DEBUG: Switch parsing - found default case" << std::endl;
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
          
          // Process the default statement's sub-statement
          if (auto substmt = defaultStmt->getSubStmt()) {
            if (auto converted = convertStatement(substmt, context)) {
              currentCase.push_back(std::move(converted));
            }
          }
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertReturnStatement(
    const clang::ReturnStmt *returnStmt, clang::ASTContext &context) {
  std::cout << "Converting return statement" << std::endl;
  
  // Handle return value if present
  std::unique_ptr<Expression> returnExpr = nullptr;
  if (auto retVal = returnStmt->getRetValue()) {
    returnExpr = convertExpression(retVal, context);
  }
  
  return std::make_unique<ReturnStmt>(std::move(returnExpr));
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
  } else if (auto unaryOp = clang::dyn_cast<clang::UnaryOperator>(expr)) {
    return convertUnaryExpression(unaryOp, context);
  } else {
    std::cout << "Unsupported expression type: " << expr->getStmtClassName()
              << std::endl;
    return nullptr;
  }
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertUnaryExpression(const clang::UnaryOperator *unaryOp,
                                            clang::ASTContext &context) {
  auto operand = convertExpression(unaryOp->getSubExpr(), context);
  if (!operand) {
    std::cout << "Failed to convert unary operand" << std::endl;
    return nullptr;
  }

  // Map Clang unary operator to interpreter unary operator
  switch (unaryOp->getOpcode()) {
    case clang::UO_PreInc:  // ++i
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::PreIncrement);
    case clang::UO_PostInc: // i++
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::PostIncrement);
    case clang::UO_PreDec:  // --i
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::PreDecrement);
    case clang::UO_PostDec: // i--
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::PostDecrement);
    case clang::UO_Plus:    // +expr
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::Plus);
    case clang::UO_Minus:   // -expr
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::Minus);
    case clang::UO_Not:     // !expr
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::LogicalNot);
    case clang::UO_LNot:    // !expr (logical not)
      return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::LogicalNot);
    default:
      std::cout << "Unsupported unary operator: " << unaryOp->getOpcodeStr(unaryOp->getOpcode()).str() << std::endl;
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
    } else if (funcName == "WaveActiveBitAnd" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::And);
      }
    } else if (funcName == "WaveActiveBitOr" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or);
      }
    } else if (funcName == "WaveActiveBitXor" && callExpr->getNumArgs() == 1) {
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
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::AllEqual);
      }
    } else if (funcName == "WaveActiveAllTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::AllTrue);
      }
    } else if (funcName == "WaveActiveAnyTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::AnyTrue);
      }
    } else if (funcName == "WaveActiveBallot" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Ballot);
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
  case clang::BO_LT:
    opType = BinaryOpExpr::Lt;
    break;
  case clang::BO_LE:
    opType = BinaryOpExpr::Le;
    break;
  case clang::BO_GT:
    opType = BinaryOpExpr::Gt;
    break;
  case clang::BO_GE:
    opType = BinaryOpExpr::Ge;
    break;
  case clang::BO_EQ:
    opType = BinaryOpExpr::Eq;
    break;
  case clang::BO_NE:
    opType = BinaryOpExpr::Ne;
    break;
  case clang::BO_LAnd:
    opType = BinaryOpExpr::And;
    break;
  case clang::BO_LOr:
    opType = BinaryOpExpr::Or;
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
        // Create a buffer access expression for global buffers (RWBuffer, etc.)
        std::cout << "Creating BufferAccessExpr for buffer: " << bufferName << std::endl;
        return std::make_unique<BufferAccessExpr>(bufferName, std::move(indexExpr));
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

// WhileStmt implementation
WhileStmt::WhileStmt(std::unique_ptr<Expression> cond,
                     std::vector<std::unique_ptr<Statement>> body)
    : condition_(std::move(cond)), body_(std::move(body)) {}

void WhileStmt::execute(LaneContext &lane, WaveContext &wave,
                        ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Find our entry in the execution stack (if any)
  int ourStackIndex = -1;
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    if (lane.executionStack[i].statement == static_cast<const void*>(this)) {
      ourStackIndex = i;
      break;
    }
  }
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " " 
            << (isResuming ? "resuming" : "starting") << " while loop" << std::endl;

  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    
    // Get current block before entering loop
    uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

    // Get current merge stack for block creation
    std::vector<MergeStackEntry> currentMergeStack =
        tg.getCurrentMergeStack(wave.waveId, lane.laneId);

    // Create loop blocks (header, merge) - pass current execution path
    auto [headerBlockId, mergeBlockId] =
        tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                            currentMergeStack, lane.executionPath);

    // Push merge point for loop divergence
    std::set<uint32_t> divergentBlocks = {headerBlockId};
    tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                      parentBlockId, divergentBlocks);

    // Move to loop header block
    tg.assignLaneToBlock(wave.waveId, lane.laneId, headerBlockId);
    tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);

    // Push execution state onto stack
    LaneContext::BlockExecutionState newState(static_cast<const void*>(this), 
                                               LaneContext::ControlFlowPhase::EvaluatingCondition, 
                                               0);
    newState.loopIteration = 0;
    newState.loopHeaderBlockId = headerBlockId;
    newState.loopMergeBlockId = mergeBlockId;
    
    lane.executionStack.push_back(newState);
    ourStackIndex = lane.executionStack.size() - 1;
  }

  // Get our execution state
  auto& ourEntry = lane.executionStack[ourStackIndex];
  uint32_t headerBlockId = ourEntry.loopHeaderBlockId;
  uint32_t mergeBlockId = ourEntry.loopMergeBlockId;

  try {
    // State machine for while loop execution
    switch (ourEntry.phase) {
      case LaneContext::ControlFlowPhase::EvaluatingCondition: {
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " evaluating condition for iteration " 
                  << ourEntry.loopIteration << std::endl;
        
        // Check loop condition
        bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
        if (!shouldContinue) {
          // Lane is exiting loop - move to reconverging phase
          ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
          break;
        }
        
        // Condition passed, move to body execution
        ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
        lane.executionStack[ourStackIndex].statementIndex = 0;
        break;
      }
      
      case LaneContext::ControlFlowPhase::ExecutingBody: {
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing body for iteration " 
                  << ourEntry.loopIteration << " at statement " << ourEntry.statementIndex << std::endl;
        
        // Execute statements from where we left off
        for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
          lane.executionStack[ourStackIndex].statementIndex = i;
          body_[i]->execute(lane, wave, tg);
          
          if (lane.hasReturned) {
            // Pop our entry and return from loop
            lane.executionStack.pop_back();
            tg.popMergePoint(wave.waveId, lane.laneId);
            return;
          }
        }
        
        // Completed body execution - move to next iteration
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " completed iteration " 
                  << lane.executionStack[ourStackIndex].loopIteration << std::endl;
        lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
        lane.executionStack[ourStackIndex].loopIteration++;
        lane.executionStack[ourStackIndex].statementIndex = 0;
        break;
      }
      
      case LaneContext::ControlFlowPhase::Reconverging: {
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " reconverging from while loop" << std::endl;
        
        // Pop our entry from execution stack
        lane.executionStack.pop_back();
        
        // Pop merge point and move to merge block
        tg.popMergePoint(wave.waveId, lane.laneId);
        tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
        tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
        
        // Loop has completed, restore active state
        lane.isActive = lane.isActive && !lane.hasReturned;
        return;
      }
      
      default:
        std::cout << "ERROR: WhileStmt - Unexpected phase " << static_cast<int>(ourEntry.phase) << std::endl;
        return;
    }
  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - execution state is already saved
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " waiting for wave operation in phase " 
              << static_cast<int>(ourEntry.phase) << " at statement " << ourEntry.statementIndex 
              << ", iteration " << ourEntry.loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      // Break - exit loop
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " breaking from while loop" << std::endl;
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      
      // Clean up - remove from blocks this lane will never reach
      tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);
      
      tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
      tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      // Continue - go to next iteration
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " continuing while loop" << std::endl;
      auto& ourEntry = lane.executionStack[ourStackIndex];
      lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
      lane.executionStack[ourStackIndex].loopIteration++;
      lane.executionStack[ourStackIndex].statementIndex = 0;
    }
  }
  
  // If we reach here, continue the loop by recursively calling execute
  // This handles the state machine transitions
  try {
    execute(lane, wave, tg);
  } catch (const WaveOperationWaitException& e) {
    // Propagate wave operation wait to parent
    throw;
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

  // Find our entry in the execution stack (if any)
  int ourStackIndex = -1;
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    if (lane.executionStack[i].statement == static_cast<const void*>(this)) {
      ourStackIndex = i;
      break;
    }
  }
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " " 
            << (isResuming ? "resuming" : "starting") << " do-while loop" << std::endl;

  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    
    // Get current block before entering loop
    uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

    // Get current merge stack for block creation
    std::vector<MergeStackEntry> currentMergeStack =
        tg.getCurrentMergeStack(wave.waveId, lane.laneId);

    // Create loop blocks (header, merge) - pass current execution path
    // For do-while, header is where condition is checked
    auto [headerBlockId, mergeBlockId] =
        tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                            currentMergeStack, lane.executionPath);

    // Push merge point for loop divergence
    std::set<uint32_t> divergentBlocks = {headerBlockId};
    tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                      parentBlockId, divergentBlocks);

    // Push execution state onto stack - DoWhile starts with body execution
    LaneContext::BlockExecutionState newState(static_cast<const void*>(this), 
                                               LaneContext::ControlFlowPhase::ExecutingBody, 
                                               0);
    newState.loopIteration = 0;
    newState.loopHeaderBlockId = headerBlockId;
    newState.loopMergeBlockId = mergeBlockId;
    
    lane.executionStack.push_back(newState);
    ourStackIndex = lane.executionStack.size() - 1;
  }

  // Get our execution state
  auto& ourEntry = lane.executionStack[ourStackIndex];
  uint32_t headerBlockId = ourEntry.loopHeaderBlockId;
  uint32_t mergeBlockId = ourEntry.loopMergeBlockId;

  try {
    // State machine for do-while loop execution
    switch (ourEntry.phase) {
      case LaneContext::ControlFlowPhase::ExecutingBody: {
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " executing body for iteration " 
                  << ourEntry.loopIteration << " at statement " << ourEntry.statementIndex << std::endl;
        
        // Execute statements from where we left off
        for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
          lane.executionStack[ourStackIndex].statementIndex = i;
          body_[i]->execute(lane, wave, tg);
          
          if (lane.hasReturned) {
            // Pop our entry and return from loop
            lane.executionStack.pop_back();
            tg.popMergePoint(wave.waveId, lane.laneId);
            return;
          }
        }
        
        // Completed body execution - move to condition evaluation
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " completed body for iteration " 
                  << lane.executionStack[ourStackIndex].loopIteration << std::endl;
        lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
        break;
      }
      
      case LaneContext::ControlFlowPhase::EvaluatingCondition: {
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " evaluating condition after iteration " 
                  << ourEntry.loopIteration << std::endl;
        
        // Check loop condition
        bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
        if (!shouldContinue) {
          // Lane is exiting loop - move to reconverging phase
          ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
          break;
        }
        
        // Condition passed, move to next iteration body execution
        lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::ExecutingBody;
        lane.executionStack[ourStackIndex].loopIteration++;
        lane.executionStack[ourStackIndex].statementIndex = 0;
        break;
      }
      
      case LaneContext::ControlFlowPhase::Reconverging: {
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " reconverging from do-while loop" << std::endl;
        
        // Pop our entry from execution stack
        lane.executionStack.pop_back();
        
        // Pop merge point and move to merge block
        tg.popMergePoint(wave.waveId, lane.laneId);
        tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
        tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
        
        // Loop has completed, restore active state
        lane.isActive = lane.isActive && !lane.hasReturned;
        return;
      }
      
      default:
        std::cout << "ERROR: DoWhileStmt - Unexpected phase " << static_cast<int>(ourEntry.phase) << std::endl;
        return;
    }
  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - execution state is already saved
    std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " waiting for wave operation in phase " 
              << static_cast<int>(ourEntry.phase) << " at statement " << ourEntry.statementIndex 
              << ", iteration " << ourEntry.loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      // Break - exit loop
      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " breaking from do-while loop" << std::endl;
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      
      // Clean up - remove from blocks this lane will never reach
      tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);
      
      tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
      tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      // Continue - go to condition evaluation
      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " continuing do-while loop" << std::endl;
      auto& ourEntry = lane.executionStack[ourStackIndex];
      lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
    }
  }
  
  // If we reach here, continue the loop by recursively calling execute
  // This handles the state machine transitions
  try {
    execute(lane, wave, tg);
  } catch (const WaveOperationWaitException& e) {
    // Propagate wave operation wait to parent
    throw;
  }
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
  std::cout << "DEBUG: SwitchStmt::addCase - adding case " << value << std::endl;
  cases_.push_back({value, std::move(stmts)});
}

void SwitchStmt::addDefault(std::vector<std::unique_ptr<Statement>> stmts) {
  std::cout << "DEBUG: SwitchStmt::addDefault - adding default case" << std::endl;
  cases_.push_back({std::nullopt, std::move(stmts)});
}

void SwitchStmt::execute(LaneContext &lane, WaveContext &wave,
                         ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Find our entry in the execution stack (if any)
  int ourStackIndex = -1;
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    if (lane.executionStack[i].statement == static_cast<const void*>(this)) {
      ourStackIndex = i;
      break;
    }
  }
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " " 
            << (isResuming ? "resuming" : "starting") << " switch statement" << std::endl;

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(static_cast<const void*>(this), 
                                     LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " starting fresh execution (pushed to stack depth=" 
              << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
    
    // Create blocks for all cases
    uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
    
    // Extract case values and check for default
    std::vector<int> caseValues;
    bool hasDefault = false;
    for (const auto& caseBlock : cases_) {
      if (caseBlock.value.has_value()) {
        caseValues.push_back(caseBlock.value.value());
      } else {
        hasDefault = true;
      }
    }
    
    // Create switch case blocks and store in execution stack
    auto& ourEntry = lane.executionStack[ourStackIndex];
    ourEntry.switchCaseBlockIds = tg.createSwitchCaseBlocks(static_cast<const void*>(this), parentBlockId,
                                                           currentMergeStack, caseValues, hasDefault, 
                                                           lane.executionPath);
    
    std::cout << "DEBUG: SwitchStmt - Created " << ourEntry.switchCaseBlockIds.size() << " case blocks" << std::endl;
  } else {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " resuming execution (found at stack index=" 
              << ourStackIndex << ", current stack depth=" << lane.executionStack.size() << ", this=" << this << ")" << std::endl;
  }
  
  try {
    while (lane.isActive) {
      auto& ourEntry = lane.executionStack[ourStackIndex];
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " in phase " << LaneContext::getPhaseString(ourEntry.phase) 
                << " (stack depth=" << lane.executionStack.size() << ", our index=" << ourStackIndex << ", this=" << this << ")" << std::endl;
      
      switch (ourEntry.phase) {
        case LaneContext::ControlFlowPhase::EvaluatingCondition: {
          std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " evaluating switch condition" << std::endl;
          
          // Only evaluate condition if not already evaluated
          if (!ourEntry.conditionEvaluated) {
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " evaluating condition for first time" << std::endl;
            auto condValue = condition_->evaluate(lane, wave, tg);
            ourEntry.switchValue = condValue;
            ourEntry.conditionEvaluated = true;
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " switch condition evaluated to: " << condValue.asInt() << std::endl;
          } else {
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " using cached condition result=" << ourEntry.switchValue.asInt() << std::endl;
          }
          
          // Find which case this lane should execute
          int switchValue = ourEntry.switchValue.asInt();
          size_t matchingCaseIndex = SIZE_MAX;
          
          for (size_t i = 0; i < cases_.size(); ++i) {
            if (cases_[i].value.has_value() && cases_[i].value.value() == switchValue) {
              matchingCaseIndex = i;
              std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                        << " matched case " << cases_[i].value.value() << " at index " << i << std::endl;
              break;
            } else if (!cases_[i].value.has_value()) {
              // Default case - only use if no exact match found
              if (matchingCaseIndex == SIZE_MAX) {
                matchingCaseIndex = i;
                std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                          << " matched default case at index " << i << std::endl;
              }
            }
          }

          if (matchingCaseIndex == SIZE_MAX) {
            // No matching case - exit switch
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " no matching case found" << std::endl;
            lane.executionStack.pop_back();
            return;
          }
          
          // Set up for case execution and move to appropriate block
          ourEntry.caseIndex = matchingCaseIndex;
          ourEntry.statementIndex = 0;
          ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingCase;
          
          // Move lane to the appropriate case block and remove from previous cases
          if (matchingCaseIndex < ourEntry.switchCaseBlockIds.size()) {
            uint32_t chosenCaseBlockId = ourEntry.switchCaseBlockIds[matchingCaseIndex];
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                      << " moving to case block " << chosenCaseBlockId << " for case " << matchingCaseIndex << std::endl;
            
            // Move to the chosen case block
            tg.moveThreadFromUnknownToParticipating(chosenCaseBlockId, wave.waveId, lane.laneId);
            
            // Remove from all previous case blocks (cases before the matching one)
            // because those cases won't be executed
            for (size_t i = 0; i < matchingCaseIndex; ++i) {
              uint32_t previousCaseBlockId = ourEntry.switchCaseBlockIds[i];
              std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                        << " removing from previous case block " << previousCaseBlockId << " (case " << i << ")" << std::endl;
              tg.removeThreadFromUnknown(previousCaseBlockId, wave.waveId, lane.laneId);
              tg.removeThreadFromNestedBlocks(previousCaseBlockId, wave.waveId, lane.laneId);
            }
            
            // Note: Lane remains in subsequent case blocks due to potential fallthrough
          }
          break;
        }
        
        case LaneContext::ControlFlowPhase::ExecutingCase: {
          size_t caseIdx = ourEntry.caseIndex;
          size_t stmtIdx = ourEntry.statementIndex;
          
          if (caseIdx >= cases_.size()) {
            // All cases completed
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " completed all cases" << std::endl;
            lane.executionStack.pop_back();
            return;
          }
          
          const auto &caseBlock = cases_[caseIdx];
          std::string caseLabel = caseBlock.value.has_value() ? 
                                  std::to_string(caseBlock.value.value()) : "default";
          
          if (stmtIdx >= caseBlock.statements.size()) {
            // Current case completed - move to next case (fallthrough)
            std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                      << " completed case " << caseLabel << ", falling through to next" << std::endl;
            
            size_t nextCaseIndex = ourEntry.caseIndex + 1;
            
            // Move lane to next case block if it exists
            if (nextCaseIndex < ourEntry.switchCaseBlockIds.size()) {
              uint32_t currentCaseBlockId = ourEntry.switchCaseBlockIds[ourEntry.caseIndex];
              uint32_t nextCaseBlockId = ourEntry.switchCaseBlockIds[nextCaseIndex];
              
              std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                        << " moving from case block " << currentCaseBlockId 
                        << " to case block " << nextCaseBlockId << " (fallthrough)" << std::endl;
              
              // Remove from current case block
              tg.removeThreadFromAllSets(currentCaseBlockId, wave.waveId, lane.laneId);
              
              // Move to next case block
              tg.moveThreadFromUnknownToParticipating(nextCaseBlockId, wave.waveId, lane.laneId);
            }
            
            ourEntry.caseIndex++;
            ourEntry.statementIndex = 0;
            break;
          }
          
          std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId 
                    << " executing statement " << stmtIdx << " in case " << caseLabel << std::endl;
          
          // Execute the current statement
          caseBlock.statements[stmtIdx]->execute(lane, wave, tg);
          
          if (!lane.isActive) {
            lane.executionStack.pop_back();
            return;
          }
          
          // Move to next statement
          ourEntry.statementIndex++;
          break;
        }
        
        default:
          std::cout << "ERROR: SwitchStmt - Lane " << lane.laneId << " unexpected phase" << std::endl;
          lane.executionStack.pop_back();
          return;
      }
    }
    
  } catch (const WaveOperationWaitException&) {
    // Wave operation is waiting - execution state is already saved
    auto& ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " waiting for wave operation in case " 
              << ourEntry.caseIndex << " at statement " << ourEntry.statementIndex << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      // Break - exit switch
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " breaking from switch" << std::endl;
      lane.executionStack.pop_back();
      return;
    }
    // Continue statements don't apply to switch - just ignore them
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " ignoring continue in switch" << std::endl;
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
  // Don't move lanes that are waiting for wave operations or barriers
  if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
    auto state = waves[waveId]->lanes[laneId]->state;
    if (state == ThreadState::WaitingForWave || state == ThreadState::WaitingAtBarrier) {
      std::cout << "DEBUG: Preventing move of waiting lane " << laneId << " (state=" << (int)state << ") to block " << blockId << std::endl;
      return;
    }
  }
  
  // Remove lane from its current block's arrived lanes first
  auto currentBlockIt = waves[waveId]->laneToCurrentBlock.find(laneId);
  if (currentBlockIt != waves[waveId]->laneToCurrentBlock.end()) {
    uint32_t oldBlockId = currentBlockIt->second;
    auto oldBlockIt = executionBlocks.find(oldBlockId);
    if (oldBlockIt != executionBlocks.end()) {
      // NEW: Don't remove from arrivedLanes if moving from header to iteration block
      // This keeps lanes in header's arrivedLanes until they exit the loop entirely
      auto newBlockIt = executionBlocks.find(blockId);
      bool isHeaderToLoopBody = (oldBlockIt->second.getBlockType() == BlockType::LOOP_HEADER && 
                                 newBlockIt != executionBlocks.end() &&
                                 newBlockIt->second.getBlockType() == BlockType::LOOP_BODY);
      
      std::cout << "DEBUG: assignLaneToBlock - moving lane " << laneId 
                << " from block " << oldBlockId << " (type " << (int)oldBlockIt->second.getBlockType() 
                << ") to block " << blockId << " (type " << (int)newBlockIt->second.getBlockType() 
                << "), isHeaderToLoopBody=" << isHeaderToLoopBody << std::endl;
      
      if (!isHeaderToLoopBody) {
        oldBlockIt->second.removeArrivedLane(waveId, laneId);
        std::cout << "DEBUG: Removed lane " << laneId << " from arrivedLanes of block " << oldBlockId << std::endl;
      } else {
        std::cout << "DEBUG: Keeping lane " << laneId << " in arrivedLanes of header block " << oldBlockId << std::endl;
      }
    }
  }
  
  waves[waveId]->laneToCurrentBlock[laneId] = blockId;

  // Add lane to the block's participating lanes
  auto it = executionBlocks.find(blockId);
  if (it != executionBlocks.end()) {
    std::cout << "DEBUG: assignLaneToBlock - adding lane " << laneId << " to block " << blockId << std::endl;
    it->second.addParticipatingLane(waveId, laneId);
    // Also add to arrived lanes when assigned
    it->second.addArrivedLane(waveId, laneId);
    auto partLanes = it->second.getParticipatingLanes();
    size_t laneCount = partLanes.count(waveId) ? partLanes.at(waveId).size() : 0;
    std::cout << "DEBUG: assignLaneToBlock - block " << blockId << " now has " 
              << laneCount << " participating lanes" << std::endl;

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
  
  std::cout << "DEBUG: findOrCreateBlockForPath called with " << unknownLanes.size() << " waves of unknown lanes" << std::endl;
  for (const auto& [waveId, laneSet] : unknownLanes) {
    std::cout << "  Wave " << waveId << ": {";
    for (LaneId laneId : laneSet) {
      std::cout << laneId << " ";
    }
    std::cout << "} (" << laneSet.size() << " lanes)" << std::endl;
  }
  
  // Check if block with this identity already exists
  auto it = identityToBlockId.find(identity);
  if (it != identityToBlockId.end()) {
    // Block exists - don't modify its unknown lanes!
    // The existing block already has the correct unknown lanes from when it was first created.
    // Those lanes will be properly removed by removeThreadFromNestedBlocks when appropriate.
    uint32_t existingBlockId = it->second;
    std::cout << "DEBUG: Found existing block " << existingBlockId 
              << " - not modifying unknown lanes" << std::endl;
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
      std::cout << "DEBUG: addUnknownLane - adding lane " << laneId 
                << " to new block " << newBlockId << std::endl;
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
    const void *sourceStmt, BlockType blockType, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack, bool conditionValue, 
    const std::vector<const void*>& executionPath) const {
  BlockIdentity identity;
  identity.sourceStatement = sourceStmt;
  identity.blockType = blockType;
  identity.conditionValue = conditionValue;
  identity.parentBlockId = parentBlockId;
  identity.executionPath = executionPath;
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

  // Use the wave-specific approach for instruction synchronization
  bool canExecuteInBlock =
      canExecuteWaveInstructionInBlock(currentBlockId, waveId, instrIdentity);
  
  // Debug unknown lanes state
  auto blockIt = executionBlocks.find(currentBlockId);
  if (blockIt != executionBlocks.end()) {
    auto unknownLanes = blockIt->second.getUnknownLanesForWave(waveId);
    std::cout << "DEBUG: Block " << currentBlockId << " wave " << waveId << " unknown lanes: {";
    for (LaneId uid : unknownLanes) {
      std::cout << uid << " ";
    }
    std::cout << "}" << std::endl;
  }

  // Check using compound key for the new system
  std::pair<const void*, uint32_t> instructionKey = {instruction, currentBlockId};
  bool allParticipantsKnown = areAllParticipantsKnownForWaveInstruction(waveId, instructionKey);
  bool allParticipantsArrived = haveAllParticipantsArrivedAtWaveInstruction(waveId, instructionKey);
  bool canExecuteGlobal = allParticipantsKnown && allParticipantsArrived;

  std::cout << "DEBUG: canExecuteWaveInstruction for lane " << laneId << " in block " << currentBlockId 
            << ": canExecuteInBlock=" << canExecuteInBlock 
            << ", allParticipantsKnown=" << allParticipantsKnown 
            << ", allParticipantsArrived=" << allParticipantsArrived
            << ", canExecuteGlobal=" << canExecuteGlobal << std::endl;

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

  // Create compound key using existing currentBlockId
  std::pair<const void*, uint32_t> instructionKey = {instruction, currentBlockId};
  
  auto &syncPoint = waves[waveId]->activeSyncPoints[instructionKey];
  syncPoint.arrivedParticipants.insert(laneId);

  // Mark lane as waiting at this instruction with compound key
  waves[waveId]->laneWaitingAtInstruction[laneId] = instructionKey;
  waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;

  // Update completion status
  syncPoint.allParticipantsArrived =
      (syncPoint.arrivedParticipants == syncPoint.expectedParticipants);
  syncPoint.isComplete =
      syncPoint.allParticipantsKnown && syncPoint.allParticipantsArrived;
}

bool ThreadgroupContext::areAllParticipantsKnownForWaveInstruction(
    WaveId waveId, const std::pair<const void*, uint32_t>& instructionKey) const {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    // No sync point created yet - check if only one lane is in the block
    uint32_t blockId = instructionKey.second;
    auto blockIt = executionBlocks.find(blockId);
    if (blockIt != executionBlocks.end()) {
      auto arrivedLanes = blockIt->second.getArrivedLanesForWave(waveId);
      if (arrivedLanes.size() == 1) {
        return true; // Single lane can proceed immediately
      }
    }
    return false;
  }

  return it->second.allParticipantsKnown;
}

bool ThreadgroupContext::haveAllParticipantsArrivedAtWaveInstruction(
    WaveId waveId, const std::pair<const void*, uint32_t>& instructionKey) const {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    return false; // No sync point created yet
  }

  return it->second.allParticipantsArrived;
}

// getWaveInstructionParticipants removed - functionality replaced by compound key system

void ThreadgroupContext::createOrUpdateWaveSyncPoint(
    const void *instruction, WaveId waveId, LaneId laneId,
    const std::string &instructionType) {
  // Create compound key with current block ID
  uint32_t blockId = getCurrentBlock(waveId, laneId);
  std::pair<const void*, uint32_t> instructionKey = {instruction, blockId};
  
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);

  if (it == waves[waveId]->activeSyncPoints.end()) {
    // Create new sync point
    WaveOperationSyncPoint syncPoint;
    syncPoint.instruction = instruction;
    syncPoint.instructionType = instructionType;

    // Use blockId from compound key
    syncPoint.blockId = blockId;

    // Get expected participants from the block
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }

    // Check if all participants are known (no unknown lanes for this wave in block)
    auto blockIt = executionBlocks.find(blockId);
    syncPoint.allParticipantsKnown = blockIt != executionBlocks.end() && 
                                     blockIt->second.areAllUnknownLanesResolvedForWave(waveId);

    waves[waveId]->activeSyncPoints[instructionKey] = syncPoint;
  } else {
    // Update existing sync point
    auto &syncPoint = it->second;

    // Update participants knowledge if block resolved more lanes for this wave
    uint32_t blockId = getCurrentBlock(waveId, laneId);
    auto blockIt = executionBlocks.find(blockId);
    syncPoint.allParticipantsKnown = blockIt != executionBlocks.end() && 
                                     blockIt->second.areAllUnknownLanesResolvedForWave(waveId);

    // Update expected participants if needed
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }
  }
}

void ThreadgroupContext::releaseWaveSyncPoint(WaveId waveId,
                                              const std::pair<const void*, uint32_t>& instructionKey) {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it != waves[waveId]->activeSyncPoints.end()) {
    const auto &syncPoint = it->second;

    INTERPRETER_DEBUG_LOG("Releasing sync point: " << syncPoint.arrivedParticipants.size() 
                         << " participants\n");

    // Release all waiting lanes
    for (LaneId laneId : syncPoint.arrivedParticipants) {
      if (laneId < waves[waveId]->lanes.size()) {
        INTERPRETER_DEBUG_LOG("  Waking up lane " << laneId 
                             << " (was " << (waves[waveId]->lanes[laneId]->state == ThreadState::WaitingForWave ? "WaitingForWave" : "other") 
                             << ")\n");
        waves[waveId]->lanes[laneId]->state = ThreadState::Ready;
        waves[waveId]->lanes[laneId]->isResumingFromWaveOp = true;  // Set resuming flag
      }
      waves[waveId]->laneWaitingAtInstruction.erase(laneId);
    }

    // Remove the sync point
    waves[waveId]->activeSyncPoints.erase(it);
    INTERPRETER_DEBUG_LOG("  Sync point removed from active list\n");
  } else {
    INTERPRETER_DEBUG_LOG("WARNING: Sync point not found for instruction " << instructionKey.first << " block " << instructionKey.second << "\n");
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

bool ThreadgroupContext::canExecuteWaveInstructionInBlock(
    uint32_t blockId, WaveId waveId, const InstructionIdentity &instruction) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return false; // Block doesn't exist
  }

  const DynamicExecutionBlock &block = blockIt->second;

  // For wave operations, only check if unknown lanes for this specific wave are resolved
  if (!block.areAllUnknownLanesResolvedForWave(waveId)) {
    return false; // Still have unknown lanes for this wave
  }

  return true; // Wave operation can proceed when all lanes from this wave are known
}

bool ThreadgroupContext::canExecuteBarrierInstructionInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return false; // Block doesn't exist
  }

  const DynamicExecutionBlock &block = blockIt->second;

  // For barriers, check if all unknown lanes across all waves are resolved
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

bool ThreadgroupContext::canExecuteInstructionInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  // Legacy function - delegate to barrier logic for backward compatibility
  return canExecuteBarrierInstructionInBlock(blockId, instruction);
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
std::tuple<uint32_t, uint32_t, uint32_t> ThreadgroupContext::createIfBlocks(
    const void *ifStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack, bool hasElse,
    const std::vector<const void*>& executionPath) {
  // Get all lanes that could potentially take either path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Always create the then block
  BlockIdentity thenIdentity =
      createBlockIdentity(ifStmt, BlockType::BRANCH_THEN, parentBlockId, mergeStack, true, executionPath);
  uint32_t thenBlockId =
      findOrCreateBlockForPath(thenIdentity, allPotentialLanes);

  uint32_t elseBlockId = 0; // 0 means no else block

  // Only create else block if it exists in the code
  if (hasElse) {
    BlockIdentity elseIdentity =
        createBlockIdentity(ifStmt, BlockType::BRANCH_ELSE, parentBlockId, mergeStack, false, executionPath);
    elseBlockId = findOrCreateBlockForPath(elseIdentity, allPotentialLanes);
  }

  // Always create merge block for reconvergence
  BlockIdentity mergeIdentity =
      createBlockIdentity(ifStmt, BlockType::MERGE, parentBlockId, mergeStack, true, executionPath);
  uint32_t mergeBlockId =
      findOrCreateBlockForPath(mergeIdentity, allPotentialLanes);

  return {thenBlockId, elseBlockId, mergeBlockId};
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
  std::cout << "DEBUG: removeThreadFromUnknown - removing lane " << laneId 
            << " from block " << blockId << std::endl;
  
  // Show unknown lanes before removal
  auto unknownBefore = block.getUnknownLanesForWave(waveId);
  std::cout << "DEBUG: Block " << blockId << " unknown lanes before removal: {";
  for (auto it = unknownBefore.begin(); it != unknownBefore.end(); ++it) {
    if (it != unknownBefore.begin()) std::cout << " ";
    std::cout << *it;
  }
  std::cout << "}" << std::endl;
  
  block.removeUnknownLane(waveId, laneId);
  
  // Show unknown lanes after removal
  auto unknownAfter = block.getUnknownLanesForWave(waveId);
  std::cout << "DEBUG: Block " << blockId << " unknown lanes after removal: {";
  for (auto it = unknownAfter.begin(); it != unknownAfter.end(); ++it) {
    if (it != unknownAfter.begin()) std::cout << " ";
    std::cout << *it;
  }
  std::cout << "}" << std::endl;

  // Update resolution status
  bool allResolved = block.areAllUnknownLanesResolvedForWave(waveId);
  std::cout << "DEBUG: Block " << blockId << " areAllUnknownLanesResolvedForWave(" << waveId << ") = " << allResolved << std::endl;
  block.setWaveAllUnknownResolved(waveId, allResolved);
  
  // CRITICAL: Check if removing this unknown lane allows any waiting wave operations to proceed
  // This handles the case where lanes 0,1 are waiting in block 2 for a wave op,
  // and lanes 2,3 choosing the else branch resolves all unknowns
  if (block.areAllUnknownLanesResolvedForWave(waveId)) {
    std::cout << "DEBUG: Block " << blockId << " now has all unknowns resolved for wave " << waveId 
              << " - checking for ready wave operations" << std::endl;
    
    // Check all waiting lanes in this block to see if their wave operations can now proceed
    auto waitingLanes = block.getWaitingLanesForWave(waveId);
    for (LaneId waitingLaneId : waitingLanes) {
      // Check if this lane is waiting for a wave operation in this block
      if (waves[waveId]->laneWaitingAtInstruction.count(waitingLaneId)) {
        auto& instructionKey = waves[waveId]->laneWaitingAtInstruction[waitingLaneId];
        // Only process if the instruction is in this block
        if (instructionKey.second == blockId) {
          std::cout << "DEBUG: Checking if waiting lane " << waitingLaneId 
                    << " can now execute wave operation in block " << blockId << std::endl;
          
          // Re-evaluate the sync point for this instruction
          auto& syncPoint = waves[waveId]->activeSyncPoints[instructionKey];
          syncPoint.allParticipantsKnown = true;  // All unknowns are now resolved
          syncPoint.allParticipantsArrived = (syncPoint.arrivedParticipants == syncPoint.expectedParticipants);
          
          if (syncPoint.allParticipantsArrived) {
            syncPoint.isComplete = true;
            std::cout << "DEBUG: Wave operation in block " << blockId 
                      << " is now ready to execute - all participants known and arrived!" << std::endl;
          }
        }
      }
    }
  }
}

// Helper method to completely remove a lane from all sets of a specific block
void ThreadgroupContext::removeThreadFromAllSets(uint32_t blockId, WaveId waveId, LaneId laneId) {
  // CRITICAL: Don't remove lanes that are waiting for wave operations
  // They need to stay in their blocks until the wave operation completes
  if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
    auto state = waves[waveId]->lanes[laneId]->state;
    if (state == ThreadState::WaitingForWave || state == ThreadState::WaitingAtBarrier) {
      std::cout << "DEBUG: removeThreadFromAllSets - NOT removing waiting lane " << laneId 
                << " (state=" << (int)state << ") from block " << blockId << std::endl;
      return;
    }
  }
  
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt != executionBlocks.end()) {
    std::cout << "DEBUG: removeThreadFromAllSets - removing lane " << laneId 
              << " from all sets of block " << blockId << std::endl;
    auto partLanesBefore = blockIt->second.getParticipatingLanes();
    size_t laneCountBefore = partLanesBefore.count(waveId) ? partLanesBefore.at(waveId).size() : 0;
    std::cout << "DEBUG: removeThreadFromAllSets - block " << blockId << " had " 
              << laneCountBefore << " participating lanes before removal" << std::endl;
    
    blockIt->second.removeParticipatingLane(waveId, laneId);
    blockIt->second.removeArrivedLane(waveId, laneId);
    blockIt->second.removeWaitingLane(waveId, laneId);
    blockIt->second.removeUnknownLane(waveId, laneId);
    
    // Update unknown resolution status after removing unknown lane
    bool allResolved = blockIt->second.areAllUnknownLanesResolvedForWave(waveId);
    blockIt->second.setWaveAllUnknownResolved(waveId, allResolved);
    
    // CRITICAL: Add the same wakeup logic as removeThreadFromUnknown
    // Check if removing this unknown lane allows any waiting wave operations to proceed
    if (allResolved) {
      std::cout << "DEBUG: removeThreadFromAllSets - Block " << blockId << " now has all unknowns resolved for wave " << waveId 
                << " - checking for ready wave operations" << std::endl;
      
      // Check all waiting lanes in this block to see if their wave operations can now proceed
      auto waitingLanes = blockIt->second.getWaitingLanesForWave(waveId);
      for (LaneId waitingLaneId : waitingLanes) {
        // Check if this lane is waiting for a wave operation in this block
        if (waves[waveId]->laneWaitingAtInstruction.count(waitingLaneId)) {
          auto& instructionKey = waves[waveId]->laneWaitingAtInstruction[waitingLaneId];
          // Only process if the instruction is in this block
          if (instructionKey.second == blockId) {
            std::cout << "DEBUG: removeThreadFromAllSets - Checking if waiting lane " << waitingLaneId 
                      << " can now execute wave operation in block " << blockId << std::endl;
            
            // Update the sync point to reflect that all participants are now known
            auto syncIt = waves[waveId]->activeSyncPoints.find(instructionKey);
            if (syncIt != waves[waveId]->activeSyncPoints.end()) {
              syncIt->second.allParticipantsKnown = true;
              std::cout << "DEBUG: removeThreadFromAllSets - Updated sync point allParticipantsKnown=true for instruction " 
                        << instructionKey.first << " in block " << blockId << std::endl;
            }
            
            // Check if the wave operation can now proceed
            if (canExecuteWaveInstruction(waveId, waitingLaneId, instructionKey.first)) {
              std::cout << "DEBUG: removeThreadFromAllSets - Waking up lane " << waitingLaneId 
                        << " in block " << blockId << " - wave operation can now proceed" << std::endl;
              waves[waveId]->lanes[waitingLaneId]->state = ThreadState::Ready;
              waves[waveId]->lanes[waitingLaneId]->isResumingFromWaveOp = true;
            }
          }
        }
      }
    }
    
    auto partLanesAfter = blockIt->second.getParticipatingLanes();
    size_t laneCountAfter = partLanesAfter.count(waveId) ? partLanesAfter.at(waveId).size() : 0;
    std::cout << "DEBUG: removeThreadFromAllSets - block " << blockId << " has " 
              << laneCountAfter << " participating lanes after removal" << std::endl;
  }
}

// TODO: verify the functional correctness of this method
void ThreadgroupContext::removeThreadFromNestedBlocks(uint32_t parentBlockId,
                                                      WaveId waveId,
                                                      LaneId laneId) {
  // Find all blocks that are nested within the parent block and remove the lane
  for (auto &[blockId, block] : executionBlocks) {
    if (block.getParentBlockId() == parentBlockId) {
      // Skip LOOP_EXIT/MERGE blocks - those are where lanes go when they exit the loop!
      // if (block.getBlockType() == BlockType::LOOP_EXIT || block.getBlockType() == BlockType::MERGE) {
      if (block.getBlockType() == BlockType::LOOP_EXIT) {
        std::cout << "DEBUG: removeThreadFromNestedBlocks - skipping " << blockId 
                  << " (LOOP_EXIT block where lanes should go)" << std::endl;
        continue;
      }
      
      // This is a direct child of the parent block - remove from all sets
      std::cout << "DEBUG: removeThreadFromNestedBlocks - removing lane " << laneId 
                << " from block " << blockId << " (child of " << parentBlockId << ")" << std::endl;
      removeThreadFromAllSets(blockId, waveId, laneId);

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

std::tuple<uint32_t, uint32_t> ThreadgroupContext::createLoopBlocks(
    const void *loopStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack,
    const std::vector<const void*>& executionPath) {
  // Get all lanes that could potentially enter the loop
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Create loop header block (where condition is checked) - parent is the block before loop
  BlockIdentity headerIdentity =
      createBlockIdentity(loopStmt, BlockType::LOOP_HEADER, parentBlockId, mergeStack);
  uint32_t headerBlockId =
      findOrCreateBlockForPath(headerIdentity, allPotentialLanes);

  // Note: Body blocks are no longer pre-created here since ForStmt, WhileStmt, and DoWhileStmt
  // all create unique iteration body blocks during execution

  // Create loop exit/merge block - parent should be header block (where threads reconverge after loop)
  BlockIdentity mergeIdentity =
      createBlockIdentity(loopStmt, BlockType::LOOP_EXIT, headerBlockId, mergeStack);
  uint32_t mergeBlockId =
      findOrCreateBlockForPath(mergeIdentity, allPotentialLanes);

  return {headerBlockId, mergeBlockId};
}

std::vector<uint32_t> ThreadgroupContext::createSwitchCaseBlocks(
    const void *switchStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack,
    const std::vector<int> &caseValues, bool hasDefault,
    const std::vector<const void*>& executionPath) {
  // Get all lanes that could potentially take any case path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  std::vector<uint32_t> caseBlockIds;

  // Create a block for each case value
  for (size_t i = 0; i < caseValues.size(); ++i) {
    // Create unique identity by including case value in the statement pointer
    const void* uniqueCasePtr = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(switchStmt) + caseValues[i] + 1);
    BlockIdentity caseIdentity =
        createBlockIdentity(uniqueCasePtr, BlockType::SWITCH_CASE, parentBlockId, mergeStack, true, executionPath);
    uint32_t caseBlockId =
        findOrCreateBlockForPath(caseIdentity, allPotentialLanes);
    caseBlockIds.push_back(caseBlockId);
  }

  // Create default block if it exists
  if (hasDefault) {
    BlockIdentity defaultIdentity =
        createBlockIdentity(switchStmt, BlockType::SWITCH_DEFAULT, parentBlockId, mergeStack, true, executionPath);
    uint32_t defaultBlockId =
        findOrCreateBlockForPath(defaultIdentity, allPotentialLanes);
    caseBlockIds.push_back(defaultBlockId);
  }

  return caseBlockIds;
}

// Debug and visualization methods implementation
void ThreadgroupContext::printDynamicExecutionGraph(bool verbose) const {
  INTERPRETER_DEBUG_LOG("\n=== Dynamic Execution Graph (MiniHLSL Interpreter) ===\n");
  INTERPRETER_DEBUG_LOG("Threadgroup Size: " << threadgroupSize << "\n");
  INTERPRETER_DEBUG_LOG("Wave Size: " << waveSize << "\n");
  INTERPRETER_DEBUG_LOG("Wave Count: " << waveCount << "\n");
  INTERPRETER_DEBUG_LOG("Total Dynamic Blocks: " << executionBlocks.size() << "\n");
  INTERPRETER_DEBUG_LOG("Next Block ID: " << nextBlockId << "\n\n");
  
  // Print each dynamic execution block
  for (const auto& [blockId, block] : executionBlocks) {
    printBlockDetails(blockId, verbose);
    INTERPRETER_DEBUG_LOG("\n");
  }
  
  if (verbose) {
    INTERPRETER_DEBUG_LOG("=== Wave States ===\n");
    for (uint32_t waveId = 0; waveId < waveCount; ++waveId) {
      printWaveState(waveId, verbose);
      INTERPRETER_DEBUG_LOG("\n");
    }
  }
  
  INTERPRETER_DEBUG_LOG("=== End Dynamic Execution Graph ===\n\n");
}

void ThreadgroupContext::printBlockDetails(uint32_t blockId, bool verbose) const {
  auto it = executionBlocks.find(blockId);
  if (it == executionBlocks.end()) {
    INTERPRETER_DEBUG_LOG("Block " << blockId << ": NOT FOUND\n");
    return;
  }
  
  const auto& block = it->second;
  INTERPRETER_DEBUG_LOG("Dynamic Block " << blockId << ":\n");
  
  // Basic block info
  INTERPRETER_DEBUG_LOG("  Block ID: " << block.getBlockId() << "\n");
  
  // Show block type from identity
  const auto& identity = block.getIdentity();
  const char* blockTypeName = "UNKNOWN";
  switch (identity.blockType) {
    case BlockType::REGULAR: blockTypeName = "REGULAR"; break;
    case BlockType::BRANCH_THEN: blockTypeName = "BRANCH_THEN"; break;
    case BlockType::BRANCH_ELSE: blockTypeName = "BRANCH_ELSE"; break;
    case BlockType::MERGE: blockTypeName = "MERGE"; break;
    case BlockType::LOOP_HEADER: blockTypeName = "LOOP_HEADER"; break;
    case BlockType::LOOP_BODY: blockTypeName = "LOOP_BODY"; break;
    case BlockType::LOOP_EXIT: blockTypeName = "LOOP_EXIT"; break;
    case BlockType::SWITCH_CASE: blockTypeName = "SWITCH_CASE"; break;
    case BlockType::SWITCH_DEFAULT: blockTypeName = "SWITCH_DEFAULT"; break;
  }
  INTERPRETER_DEBUG_LOG("  Block Type: " << blockTypeName << "\n");
  INTERPRETER_DEBUG_LOG("  Parent Block: " << block.getParentBlockId() << "\n");
  INTERPRETER_DEBUG_LOG("  Program Point: " << block.getProgramPoint() << "\n");
  INTERPRETER_DEBUG_LOG("  Is Converged: " << (block.getIsConverged() ? "Yes" : "No") << "\n");
  INTERPRETER_DEBUG_LOG("  Nesting Level: " << block.getNestingLevel() << "\n");
  
  // Source statement info
  if (block.getSourceStatement()) {
    INTERPRETER_DEBUG_LOG("  Source Statement: " << static_cast<const void*>(block.getSourceStatement()) << "\n");
  }
  
  // Participating lanes by wave
  const auto& participatingLanes = block.getParticipatingLanes();
  size_t totalLanes = 0;
  INTERPRETER_DEBUG_LOG("  Participating Lanes by Wave:\n");
  for (const auto& [waveId, laneSet] : participatingLanes) {
    INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
    bool first = true;
    for (LaneId laneId : laneSet) {
      if (!first) INTERPRETER_DEBUG_LOG(", ");
      INTERPRETER_DEBUG_LOG(laneId);
      first = false;
    }
    INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
    totalLanes += laneSet.size();
  }
  INTERPRETER_DEBUG_LOG("  Total Participating Lanes: " << totalLanes << "\n");
  
  if (verbose) {
    // Unknown lanes
    const auto& unknownLanes = block.getUnknownLanes();
    if (!unknownLanes.empty()) {
      INTERPRETER_DEBUG_LOG("  Unknown Lanes by Wave:\n");
      for (const auto& [waveId, laneSet] : unknownLanes) {
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first) INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }
    
    // Arrived lanes
    const auto& arrivedLanes = block.getArrivedLanes();
    if (!arrivedLanes.empty()) {
      INTERPRETER_DEBUG_LOG("  Arrived Lanes by Wave:\n");
      for (const auto& [waveId, laneSet] : arrivedLanes) {
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first) INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }
    
    // Waiting lanes
    const auto& waitingLanes = block.getWaitingLanes();
    if (!waitingLanes.empty()) {
      INTERPRETER_DEBUG_LOG("  Waiting Lanes by Wave:\n");
      for (const auto& [waveId, laneSet] : waitingLanes) {
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first) INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }
    
    // Instructions in this block
    const auto& instructions = block.getInstructionList();
    if (!instructions.empty()) {
      INTERPRETER_DEBUG_LOG("  Instructions (" << instructions.size() << "):\n");
      for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& instr = instructions[i];
        INTERPRETER_DEBUG_LOG("    " << i << ": " << instr.instructionType 
                            << " (ptr: " << instr.instruction << ")\n");
      }
    }
    
    // Unknown resolution status
    const auto& allUnknownResolved = block.getAllUnknownResolved();
    if (!allUnknownResolved.empty()) {
      INTERPRETER_DEBUG_LOG("  Unknown Resolution Status by Wave:\n");
      for (const auto& [waveId, resolved] : allUnknownResolved) {
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": " << (resolved ? "Resolved" : "Unresolved") << "\n");
      }
    }
  }
}

void ThreadgroupContext::printWaveState(WaveId waveId, bool verbose) const {
  if (waveId >= waves.size()) {
    INTERPRETER_DEBUG_LOG("Wave " << waveId << ": NOT FOUND\n");
    return;
  }
  
  const auto& wave = waves[waveId];
  INTERPRETER_DEBUG_LOG("Wave " << waveId << ":\n");
  INTERPRETER_DEBUG_LOG("  Wave Size: " << wave->waveSize << "\n");
  INTERPRETER_DEBUG_LOG("  Lane Count: " << wave->lanes.size() << "\n");
  INTERPRETER_DEBUG_LOG("  Active Lanes: " << wave->countActiveLanes() << "\n");
  INTERPRETER_DEBUG_LOG("  Currently Active Lanes: " << wave->countCurrentlyActiveLanes() << "\n");
  
  if (verbose) {
    // Lane to block mapping
    INTERPRETER_DEBUG_LOG("  Lane to Block Mapping:\n");
    for (const auto& [laneId, blockId] : wave->laneToCurrentBlock) {
      INTERPRETER_DEBUG_LOG("    Lane " << laneId << " -> Block " << blockId << "\n");
    }
    
    // Active wave operations
    if (!wave->activeWaveOps.empty()) {
      INTERPRETER_DEBUG_LOG("  Active Wave Operations (" << wave->activeWaveOps.size() << "):\n");
      for (const auto& [opId, waveOp] : wave->activeWaveOps) {
        INTERPRETER_DEBUG_LOG("    Op " << opId << ": Wave " << waveOp.waveId 
                            << ", Participants: " << waveOp.participatingLanes.size()
                            << ", Completed: " << waveOp.completedLanes.size() << "\n");
      }
    }
    
    // Active sync points
    if (!wave->activeSyncPoints.empty()) {
      INTERPRETER_DEBUG_LOG("  Active Sync Points (" << wave->activeSyncPoints.size() << "):\n");
      for (const auto& [instructionKey, syncPoint] : wave->activeSyncPoints) {
        INTERPRETER_DEBUG_LOG("    Instruction " << instructionKey.first << " block " << instructionKey.second << " (" << syncPoint.instructionType << "):\n");
        INTERPRETER_DEBUG_LOG("      Expected: " << syncPoint.expectedParticipants.size() << " lanes\n");
        INTERPRETER_DEBUG_LOG("      Arrived: " << syncPoint.arrivedParticipants.size() << " lanes\n");
        INTERPRETER_DEBUG_LOG("      Complete: " << (syncPoint.isComplete ? "Yes" : "No") << "\n");
      }
    }
  }
}

std::string ThreadgroupContext::getBlockSummary(uint32_t blockId) const {
  auto it = executionBlocks.find(blockId);
  if (it == executionBlocks.end()) {
    return "Block " + std::to_string(blockId) + ": NOT FOUND";
  }
  
  const auto& block = it->second;
  size_t totalLanes = 0;
  for (const auto& [waveId, laneSet] : block.getParticipatingLanes()) {
    totalLanes += laneSet.size();
  }
  
  std::stringstream ss;
  ss << "Block " << blockId << " (Parent: " << block.getParentBlockId() 
     << ", Lanes: " << totalLanes << ", Converged: " << (block.getIsConverged() ? "Y" : "N") << ")";
  return ss.str();
}

void ThreadgroupContext::printFinalVariableValues() const {
  INTERPRETER_DEBUG_LOG("\n=== Final Variable Values ===\n");
  
  for (size_t waveId = 0; waveId < waves.size(); ++waveId) {
    const auto& wave = waves[waveId];
    INTERPRETER_DEBUG_LOG("Wave " << waveId << ":\n");
    
    for (size_t laneId = 0; laneId < wave->lanes.size(); ++laneId) {
      const auto& lane = wave->lanes[laneId];
      INTERPRETER_DEBUG_LOG("  Lane " << laneId << ":\n");
      
      // Print all variables for this lane
      if (lane->variables.empty()) {
        INTERPRETER_DEBUG_LOG("    (no variables)\n");
      } else {
        for (const auto& [varName, value] : lane->variables) {
          INTERPRETER_DEBUG_LOG("    " << varName << " = " << value.toString() << "\n");
        }
      }
      
      // Print return value if present
      if (lane->hasReturned) {
        INTERPRETER_DEBUG_LOG("    (returned: " << lane->returnValue.toString() << ")\n");
      }
      
      // Print thread state
      const char* stateStr = "Unknown";
      switch (lane->state) {
        case ThreadState::Ready: stateStr = "Ready"; break;
        case ThreadState::WaitingAtBarrier: stateStr = "WaitingAtBarrier"; break;
        case ThreadState::WaitingForWave: stateStr = "WaitingForWave"; break;
        case ThreadState::Completed: stateStr = "Completed"; break;
        case ThreadState::Error: stateStr = "Error"; break;
      }
      INTERPRETER_DEBUG_LOG("    (state: " << stateStr << ")\n");
      
      // Print error message if in error state
      if (lane->state == ThreadState::Error && !lane->errorMessage.empty()) {
        INTERPRETER_DEBUG_LOG("    (error: " << lane->errorMessage << ")\n");
      }
    }
  }
  
  INTERPRETER_DEBUG_LOG("=== End Variable Values ===\n\n");
}

} // namespace interpreter
} // namespace minihlsl