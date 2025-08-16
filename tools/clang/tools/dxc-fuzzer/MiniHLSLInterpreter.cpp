#include "MiniHLSLInterpreter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

// Static member definitions
namespace minihlsl {
namespace interpreter {
std::atomic<uint32_t> Expression::nextStableId{1};
} // namespace interpreter
} // namespace minihlsl

// Clang AST includes for conversion
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"  // For HLSLSemanticAttr
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

// Debug flags for MiniHLSL Interpreter
// - ENABLE_INTERPRETER_DEBUG: Shows detailed dynamic execution block tracking
// - ENABLE_WAVE_DEBUG: Shows wave operation synchronization details
// - ENABLE_BLOCK_DEBUG: Shows block creation, merging, and convergence

static constexpr bool ENABLE_INTERPRETER_DEBUG =
    false; // Set to true to enable detailed execution tracing
static constexpr bool ENABLE_WAVE_DEBUG =
    false; // Set to true to enable wave operation tracing
static constexpr bool ENABLE_BLOCK_DEBUG =
    true; // Set to true to enable block lifecycle tracing
static constexpr bool ENABLE_PARSER_DEBUG =
    false; // Set to true to enable AST conversion debug output

#define INTERPRETER_DEBUG_LOG(msg)                                             \
  do {                                                                         \
    if (ENABLE_INTERPRETER_DEBUG) {                                            \
      llvm::errs() << msg;                                                     \
    }                                                                          \
  } while (0)
#define WAVE_DEBUG_LOG(msg)                                                    \
  do {                                                                         \
    if (ENABLE_WAVE_DEBUG) {                                                   \
      llvm::errs() << msg;                                                     \
    }                                                                          \
  } while (0)
#define BLOCK_DEBUG_LOG(msg)                                                   \
  do {                                                                         \
    if (ENABLE_BLOCK_DEBUG) {                                                  \
      llvm::errs() << msg;                                                     \
    }                                                                          \
  } while (0)

#define PARSER_DEBUG_LOG(msg)                                                  \
  do {                                                                         \
    if (ENABLE_PARSER_DEBUG) {                                                 \
      std::cout << msg << std::endl;                                          \
    }                                                                          \
  } while (0)

namespace minihlsl {
namespace interpreter {

// Forward declarations
void initializeBuiltinVariables(LaneContext& lane, 
                               WaveContext& wave,
                               ThreadgroupContext& tg,
                               const Program& program);

// Helper function to check if a thread state should be protected from
// overwriting
static bool isProtectedState(ThreadState state) {
  return state == ThreadState::WaitingForWave ||
         state == ThreadState::WaitingAtBarrier;
}

// Helper function to check if iteration marker exists anywhere in merge stack
static bool
hasIterationMarkerInStack(const std::vector<MergeStackEntry> &mergeStack,
                          const void *iterationMarker) {
  for (const auto &entry : mergeStack) {
    if (entry.sourceStatement == iterationMarker) {
      return true;
    }
  }
  return false;
}

// Helper function to set thread state if unprotected
static inline void setThreadStateIfUnprotected(LaneContext& lane) {
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

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
GlobalBuffer::GlobalBuffer(uint32_t size, const std::string &type)
    : size_(size), bufferType_(type) {
  // Initialize buffer with default values (0)
  for (uint32_t i = 0; i < size_; ++i) {
    data_[i] = Value(0);
  }
}

Value GlobalBuffer::load(uint32_t index) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer access out of bounds: "
                          << index << " >= " << size_);
    return Value(0);
  }
  return data_[index];
}

void GlobalBuffer::store(uint32_t index, const Value &value) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer access out of bounds: "
                          << index << " >= " << size_);
    return;
  }
  data_[index] = value;
}

Value GlobalBuffer::atomicAdd(uint32_t index, const Value &value) {
  if (index >= size_) {
    INTERPRETER_DEBUG_LOG("ERROR: Global buffer atomic access out of bounds: "
                          << index << " >= " << size_);
    return Value(0);
  }

  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) &&
      std::holds_alternative<int32_t>(value.data)) {
    data_[index] =
        Value(std::get<int32_t>(oldValue.data) + std::get<int32_t>(value.data));
  } else if (std::holds_alternative<float>(oldValue.data) &&
             std::holds_alternative<float>(value.data)) {
    data_[index] =
        Value(std::get<float>(oldValue.data) + std::get<float>(value.data));
  } else {
    INTERPRETER_DEBUG_LOG("ERROR: Type mismatch in atomic add operation");
  }
  return oldValue;
}

Value GlobalBuffer::atomicSub(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) &&
      std::holds_alternative<int32_t>(value.data)) {
    data_[index] =
        Value(std::get<int32_t>(oldValue.data) - std::get<int32_t>(value.data));
  } else if (std::holds_alternative<float>(oldValue.data) &&
             std::holds_alternative<float>(value.data)) {
    data_[index] =
        Value(std::get<float>(oldValue.data) - std::get<float>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicMin(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) &&
      std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::min(std::get<int32_t>(oldValue.data),
                                  std::get<int32_t>(value.data)));
  } else if (std::holds_alternative<float>(oldValue.data) &&
             std::holds_alternative<float>(value.data)) {
    data_[index] = Value(
        std::min(std::get<float>(oldValue.data), std::get<float>(value.data)));
  }
  return oldValue;
}

Value GlobalBuffer::atomicMax(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int32_t>(oldValue.data) &&
      std::holds_alternative<int32_t>(value.data)) {
    data_[index] = Value(std::max(std::get<int32_t>(oldValue.data),
                                  std::get<int32_t>(value.data)));
  } else if (std::holds_alternative<float>(oldValue.data) &&
             std::holds_alternative<float>(value.data)) {
    data_[index] = Value(
        std::max(std::get<float>(oldValue.data), std::get<float>(value.data)));
  }
  return oldValue;
}

Value GlobalBuffer::atomicAnd(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int>(oldValue.data) &&
      std::holds_alternative<int>(value.data)) {
    data_[index] =
        Value(std::get<int>(oldValue.data) & std::get<int>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicOr(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int>(oldValue.data) &&
      std::holds_alternative<int>(value.data)) {
    data_[index] =
        Value(std::get<int>(oldValue.data) | std::get<int>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicXor(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int>(oldValue.data) &&
      std::holds_alternative<int>(value.data)) {
    data_[index] =
        Value(std::get<int>(oldValue.data) ^ std::get<int>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicExchange(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  data_[index] = value;
  return oldValue;
}

Value GlobalBuffer::atomicCompareExchange(uint32_t index,
                                          const Value &compareValue,
                                          const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (oldValue.toString() == compareValue.toString()) {
    data_[index] = value;
  }
  return oldValue;
}

bool GlobalBuffer::hasConflictingAccess(uint32_t index, ThreadId tid1,
                                        ThreadId tid2) const {
  auto it = accessHistory_.find(index);
  if (it == accessHistory_.end())
    return false;

  return it->second.count(tid1) > 0 && it->second.count(tid2) > 0;
}

std::map<uint32_t, Value> GlobalBuffer::getSnapshot() const { return data_; }

void GlobalBuffer::clear() {
  data_.clear();
  accessHistory_.clear();
  for (uint32_t i = 0; i < size_; ++i) {
    data_[i] = Value(0);
  }
}

void GlobalBuffer::printContents() const {
  INTERPRETER_DEBUG_LOG("=== Global Buffer Contents ("
                        << bufferType_ << ", size=" << size_ << ") ===");
  for (const auto &[index, value] : data_) {
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
  uint32_t initialBlockId = findOrCreateBlockForPath(initialIdentity, {});

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
      INTERPRETER_DEBUG_LOG("DEBUG: Initializing wave " << w << " with "
                << allLanes.size() << " lanes in initial block "
                << initialBlockId);
      for (LaneId laneId : allLanes) {
        markLaneArrived(w, laneId, initialBlockId);
        // Verify lane was assigned
        uint32_t assignedBlock = membershipRegistry.getCurrentBlock(w, laneId);
        INTERPRETER_DEBUG_LOG("DEBUG: Lane " << laneId << " assigned to block "
                  << assignedBlock);
      }
    }
  }

  sharedMemory = std::make_shared<SharedMemory>();
}

// PHASE 2: BlockMembershipRegistry implementation
void BlockMembershipRegistry::setLaneStatus(uint32_t waveId, LaneId laneId,
                                            uint32_t blockId,
                                            LaneBlockStatus status) {
  auto key = std::make_tuple(waveId, laneId, blockId);
  if (status == LaneBlockStatus::Left) {
    membership_.erase(key);
  } else {
    membership_[key] = status;
  }
}

LaneBlockStatus BlockMembershipRegistry::getLaneStatus(uint32_t waveId,
                                                       LaneId laneId,
                                                       uint32_t blockId) const {
  auto key = std::make_tuple(waveId, laneId, blockId);
  auto it = membership_.find(key);
  return (it != membership_.end()) ? it->second : LaneBlockStatus::Unknown;
}

void BlockMembershipRegistry::registerBlock(uint32_t blockId, BlockType type) {
  blockTypes_[blockId] = type;
}

uint32_t BlockMembershipRegistry::getCurrentBlock(uint32_t waveId,
                                                  LaneId laneId) const {
  // Find all blocks where this lane is currently active
  std::vector<std::pair<uint32_t, LaneBlockStatus>> activeBlocks;
  
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<1>(key) == laneId) {
      if (status == LaneBlockStatus::Participating ||
          status == LaneBlockStatus::WaitingForWave) {
        activeBlocks.push_back({std::get<2>(key), status});
      }
    }
  }
  
  if (activeBlocks.empty()) {
    return 0;
  }
  
  if (activeBlocks.size() == 1) {
    return activeBlocks[0].first;
  }
  
  // Multiple blocks - prioritize by:
  // 1. WaitingForWave status (active synchronization)
  // 2. Non-header blocks (actual execution location)
  // 3. Highest block ID among non-headers (innermost)
  
  // First priority: blocks where lane is WaitingForWave
  for (const auto& [blockId, status] : activeBlocks) {
    if (status == LaneBlockStatus::WaitingForWave) {
      return blockId;  // This is where the lane is actively synchronizing
    }
  }
  
  // Second priority: participating in non-header blocks
  uint32_t bestNonHeaderBlock = 0;
  for (const auto& [blockId, status] : activeBlocks) {
    auto typeIt = blockTypes_.find(blockId);
    if (typeIt != blockTypes_.end() && 
        typeIt->second != BlockType::LOOP_HEADER) {
      // Prefer higher block IDs among non-headers (likely innermost)
      bestNonHeaderBlock = std::max(bestNonHeaderBlock, blockId);
    }
  }
  
  if (bestNonHeaderBlock > 0) {
    return bestNonHeaderBlock;
  }
  
  // Last resort: return highest block ID (maintains backward compatibility)
  uint32_t maxBlock = 0;
  for (const auto& [blockId, status] : activeBlocks) {
    maxBlock = std::max(maxBlock, blockId);
  }
  
  return maxBlock;
}

std::set<LaneId>
BlockMembershipRegistry::getParticipatingLanes(uint32_t waveId,
                                               uint32_t blockId) const {
  std::set<LaneId> result;
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<2>(key) == blockId &&
        status == LaneBlockStatus::Participating) {
      result.insert(std::get<1>(key));
    }
  }
  return result;
}

std::set<LaneId>
BlockMembershipRegistry::getArrivedLanes(uint32_t waveId,
                                         uint32_t blockId) const {
  // In the simplified model, "arrived" and "participating" are the same
  // This means getCurrentBlockParticipants will double-count, but that's better
  // than missing lanes
  std::set<LaneId> result;
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<2>(key) == blockId &&
        status == LaneBlockStatus::Participating) {
      result.insert(std::get<1>(key));
    }
  }
  return result;
}

std::set<LaneId>
BlockMembershipRegistry::getUnknownLanes(uint32_t waveId,
                                         uint32_t blockId) const {
  std::set<LaneId> result;
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<2>(key) == blockId &&
        status == LaneBlockStatus::Unknown) {
      result.insert(std::get<1>(key));
    }
  }
  return result;
}

std::set<LaneId>
BlockMembershipRegistry::getWaitingLanes(uint32_t waveId,
                                         uint32_t blockId) const {
  std::set<LaneId> result;
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<2>(key) == blockId &&
        status == LaneBlockStatus::WaitingForWave) {
      result.insert(std::get<1>(key));
    }
  }
  return result;
}

bool BlockMembershipRegistry::isWaveAllUnknownResolved(uint32_t waveId,
                                                       uint32_t blockId) const {
  return getUnknownLanes(waveId, blockId).empty();
}

// Convenience methods for common state transitions
void BlockMembershipRegistry::onLaneJoinBlock(uint32_t waveId, LaneId laneId,
                                              uint32_t blockId) {
  setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Participating);
}

void BlockMembershipRegistry::onLaneLeaveBlock(uint32_t waveId, LaneId laneId,
                                               uint32_t blockId) {
  setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Left);
}

void BlockMembershipRegistry::onLaneStartWaveOp(uint32_t waveId, LaneId laneId,
                                                uint32_t blockId) {
  setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::WaitingForWave);
}

void BlockMembershipRegistry::onLaneFinishWaveOp(uint32_t waveId, LaneId laneId,
                                                 uint32_t blockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: onLaneFinishWaveOp - Lane " << laneId
            << " finishing wave op in block " << blockId);

  // Check current status before update
  auto currentStatus = getLaneStatus(waveId, laneId, blockId);
  INTERPRETER_DEBUG_LOG("DEBUG: onLaneFinishWaveOp - Current status in block " << blockId
            << ": " << (int)currentStatus);

  // Only update if lane is actually in this block as WaitingForWave
  if (currentStatus == LaneBlockStatus::WaitingForWave) {
    setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Participating);
    INTERPRETER_DEBUG_LOG("DEBUG: onLaneFinishWaveOp - Updated lane " << laneId
              << " to Participating in block " << blockId);
  } else {
    INTERPRETER_DEBUG_LOG("WARNING: onLaneFinishWaveOp - Lane " << laneId
              << " not WaitingForWave in block " << blockId
              << ", skipping update");
  }
}

void BlockMembershipRegistry::onLaneReturn(uint32_t waveId, LaneId laneId) {
  // Remove lane from all blocks by setting status to Left
  std::vector<std::tuple<uint32_t, LaneId, uint32_t>> toRemove;
  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<1>(key) == laneId) {
      toRemove.push_back(key);
    }
  }
  for (const auto &key : toRemove) {
    membership_.erase(key);
  }
}

void BlockMembershipRegistry::printMembershipState() const {
  std::cout << "=== BlockMembershipRegistry State ===\n";
  for (const auto &[key, status] : membership_) {
    uint32_t waveId = std::get<0>(key);
    LaneId laneId = std::get<1>(key);
    uint32_t blockId = std::get<2>(key);
    const char *statusStr =
        (status == LaneBlockStatus::Unknown)          ? "Unknown"
        : (status == LaneBlockStatus::Participating)  ? "Participating"
        : (status == LaneBlockStatus::WaitingForWave) ? "WaitingForWave"
                                                      : "Left";
    std::cout << "  Wave " << waveId << " Lane " << laneId << " Block "
              << blockId << ": " << statusStr << "\n";
  }
  std::cout << "=====================================\n";
}

// High-level block operations that maintain consistency between registry and
// old system
void BlockMembershipRegistry::addParticipatingLaneToBlock(
    uint32_t blockId, WaveId waveId, LaneId laneId,
    std::map<uint32_t, DynamicExecutionBlock> &executionBlocks) {
  // Update registry first
  setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Participating);

  // Update old system for consistency
  // auto blockIt = executionBlocks.find(blockId);
  // if (blockIt != executionBlocks.end()) {
  //   blockIt->second.addParticipatingLane(waveId, laneId);
  // }
}

void BlockMembershipRegistry::removeParticipatingLaneFromBlock(
    uint32_t blockId, WaveId waveId, LaneId laneId,
    std::map<uint32_t, DynamicExecutionBlock> &executionBlocks) {
  // Update registry to Left status
  setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Left);

  // Update old system for consistency
  // auto blockIt = executionBlocks.find(blockId);
  // if (blockIt != executionBlocks.end()) {
  //   blockIt->second.removeParticipatingLane(waveId, laneId);
  // }
}

// Registry-based implementations of lane set getters
std::map<WaveId, std::set<LaneId>> DynamicExecutionBlock::getParticipatingLanes(
    const ThreadgroupContext &tg) const {
  std::map<WaveId, std::set<LaneId>> result;
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    auto lanes = tg.membershipRegistry.getParticipatingLanes(waveId, blockId_);
    if (!lanes.empty()) {
      result[waveId] = lanes;
    }
  }
  return result;
}

std::map<WaveId, std::set<LaneId>>
DynamicExecutionBlock::getArrivedLanes(const ThreadgroupContext &tg) const {
  std::map<WaveId, std::set<LaneId>> result;
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    auto lanes = tg.membershipRegistry.getArrivedLanes(waveId, blockId_);
    if (!lanes.empty()) {
      result[waveId] = lanes;
    }
  }
  return result;
}

std::map<WaveId, std::set<LaneId>>
DynamicExecutionBlock::getUnknownLanes(const ThreadgroupContext &tg) const {
  std::map<WaveId, std::set<LaneId>> result;
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    auto lanes = tg.membershipRegistry.getUnknownLanes(waveId, blockId_);
    if (!lanes.empty()) {
      result[waveId] = lanes;
    }
  }
  return result;
}

std::map<WaveId, std::set<LaneId>>
DynamicExecutionBlock::getWaitingLanes(const ThreadgroupContext &tg) const {
  std::map<WaveId, std::set<LaneId>> result;
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    auto lanes = tg.membershipRegistry.getWaitingLanes(waveId, blockId_);
    if (!lanes.empty()) {
      result[waveId] = lanes;
    }
  }
  return result;
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

// WaveOperationSyncPoint method implementation
bool WaveOperationSyncPoint::isAllParticipantsKnown(
    const ThreadgroupContext &tg, uint32_t waveId) const {
  const auto *block = tg.getBlock(blockId);
  if (block) {
    // Check if all unknown lanes are resolved using registry
    auto unknownLanes = tg.membershipRegistry.getUnknownLanes(waveId, blockId);
    bool result = unknownLanes.empty();

    if (!result) {
      INTERPRETER_DEBUG_LOG("DEBUG: isAllParticipantsKnown - Block " << blockId
                << " wave " << waveId << " has " << unknownLanes.size()
                << " unknown lanes: ");
      for (auto laneId : unknownLanes) {
        INTERPRETER_DEBUG_LOG(laneId << " ");
      }
      INTERPRETER_DEBUG_LOG(" - These lanes need to be resolved to Participating or Left");
    }

    return result;
  }
  return false;
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

BinaryOpExpr::BinaryOpExpr(std::unique_ptr<Expression> left,
                           std::unique_ptr<Expression> right, OpType op)
    : Expression(""), left_(std::move(left)), right_(std::move(right)), op_(op) {}

bool BinaryOpExpr::isDeterministic() const {
  return left_->isDeterministic() && right_->isDeterministic();
}

std::string BinaryOpExpr::toString() const {
  static const char *opStrings[] = {
      "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "^", "&", "|"};
  return "(" + left_->toString() + " " + opStrings[op_] + " " +
         right_->toString() + ")";
}

// AssignExpr implementation
AssignExpr::AssignExpr(const std::string& varName, std::unique_ptr<Expression> value)
    : Expression(""), varName_(varName), value_(std::move(value)) {}

Result<Value, ExecutionError>
AssignExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) const {
  // Evaluate the value expression
#ifdef _MSC_VER
  auto _result = value_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  auto val = _result.unwrap();
#else
  auto val = TRY_RESULT(value_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif
  
  INTERPRETER_DEBUG_LOG("DEBUG: AssignExpr - Lane " << lane.laneId 
                        << " assigning " << val.toString() 
                        << " to variable '" << varName_ << "'");
  
  // Perform the assignment (side effect)
  lane.variables[varName_] = val;
  
  // Call hook if available
  if (tg.interpreter) {
    tg.interpreter->onVariableAccess(lane, wave, tg, varName_, true, val);
  }
  
  // Return the assigned value (like C/C++ assignment expressions)
  return Ok<Value, ExecutionError>(val);
}

std::string AssignExpr::toString() const {
  return varName_ + " = " + value_->toString();
}

UnaryOpExpr::UnaryOpExpr(std::unique_ptr<Expression> expr, OpType op)
    : Expression(""), expr_(std::move(expr)), op_(op) {}

bool UnaryOpExpr::isDeterministic() const { return expr_->isDeterministic(); }

// Pure Result-based implementations for key expression types

Result<Value, ExecutionError>
VariableExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                              ThreadgroupContext &tg) const {
  INTERPRETER_DEBUG_LOG("DEBUG: VariableExpr - Lane " << lane.laneId
            << " evaluating variable '" << name_ << "' (Result-based)");

  auto it = lane.variables.find(name_);
  if (it == lane.variables.end()) {
    INTERPRETER_DEBUG_LOG("DEBUG: VariableExpr - Variable '" << name_ << "' not found");
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }

  INTERPRETER_DEBUG_LOG("DEBUG: VariableExpr - Variable '" << name_
            << "' = " << it->second.toString() << " (lane " << lane.laneId
            << " at " << &lane << ")");
  
  // Call hook for variable read access
  if (tg.interpreter) {
    tg.interpreter->onVariableAccess(lane, wave, tg, name_, false, it->second);
  }
  
  return Ok<Value, ExecutionError>(it->second);
}

Result<Value, ExecutionError>
BinaryOpExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                              ThreadgroupContext &tg) const {

  // Evaluate left operand
#ifdef _MSC_VER
  auto _result = left_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  auto leftVal = _result.unwrap();
#else
  auto leftVal = TRY_RESULT(left_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif

  // Evaluate right operand
#ifdef _MSC_VER
  auto _result2 = right_->evaluate_result(lane, wave, tg);
  if (_result2.is_err()) return Err<Value, ExecutionError>(_result2.unwrap_err());
  auto rightVal = _result2.unwrap();
#else
  auto rightVal = TRY_RESULT(right_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif

  // Debug output for binary operations
  std::string opStr;
  switch (op_) {
  case Add: opStr = "+"; break;
  case Sub: opStr = "-"; break;
  case Mul: opStr = "*"; break;
  case Div: opStr = "/"; break;
  case Mod: opStr = "%"; break;
  case Eq: opStr = "=="; break;
  case Ne: opStr = "!="; break;
  case Lt: opStr = "<"; break;
  case Le: opStr = "<="; break;
  case Gt: opStr = ">"; break;
  case Ge: opStr = ">="; break;
  case And: opStr = "&&"; break;
  case Or: opStr = "||"; break;
  case Xor: opStr = "^"; break;
  case BitwiseAnd: opStr = "&"; break;
  case BitwiseOr: opStr = "|"; break;
  }
  
  // Perform operation
  Value result;
  switch (op_) {
  case Add:
    result = leftVal + rightVal;
    break;
  case Sub:
    result = leftVal - rightVal;
    break;
  case Mul:
    result = leftVal * rightVal;
    break;
  case Div:
    // Could add division by zero check here
    result = leftVal / rightVal;
    break;
  case Mod:
    result = leftVal % rightVal;
    break;
  case Eq:
    result = Value(leftVal == rightVal);
    break;
  case Ne:
    result = Value(leftVal != rightVal);
    break;
  case Lt:
    result = Value(leftVal < rightVal);
    break;
  case Le:
    result = Value(leftVal <= rightVal);
    break;
  case Gt:
    result = Value(leftVal > rightVal);
    break;
  case Ge:
    result = Value(leftVal >= rightVal);
    break;
  case And:
    result = leftVal && rightVal;
    break;
  case Or:
    result = leftVal || rightVal;
    break;
  case Xor:
    result = Value(leftVal.asInt() ^ rightVal.asInt());
    break;
  case BitwiseAnd:
    result = Value(leftVal.asInt() & rightVal.asInt());
    break;
  case BitwiseOr:
    result = Value(leftVal.asInt() | rightVal.asInt());
    break;
  default:
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  
  INTERPRETER_DEBUG_LOG("DEBUG: BinaryOp - Lane " << lane.laneId << ": " 
            << leftVal.toString() << " " << opStr << " " 
            << rightVal.toString() << " = " << result.toString());
  
  return Ok<Value, ExecutionError>(result);
}

Result<Value, ExecutionError>
UnaryOpExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg) const {

#ifdef _MSC_VER
  auto _result = expr_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  auto val = _result.unwrap();
#else
  auto val = TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif

  switch (op_) {
  case Neg:
  case Minus:
    return Ok<Value, ExecutionError>(Value(-val.asFloat()));
  case Not:
  case LogicalNot:
    return Ok<Value, ExecutionError>(!val);
  case Plus:
    return Ok<Value, ExecutionError>(val);
  case PreIncrement: {
    // ++i: increment first, then return new value
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName = varExpr->toString();
      Value &var = lane.variables[varName];
      var = Value(var.asInt() + 1);
      return Ok<Value, ExecutionError>(var);
    }
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  case PostIncrement: {
    // i++: return old value, then increment
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName = varExpr->toString();
      Value &var = lane.variables[varName];
      Value oldValue = var;
      var = Value(var.asInt() + 1);
      return Ok<Value, ExecutionError>(oldValue);
    }
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  case PreDecrement: {
    // --i: decrement first, then return new value
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName = varExpr->toString();
      Value &var = lane.variables[varName];
      var = Value(var.asInt() - 1);
      return Ok<Value, ExecutionError>(var);
    }
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  case PostDecrement: {
    // i--: return old value, then decrement
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName = varExpr->toString();
      Value &var = lane.variables[varName];
      Value oldValue = var;
      var = Value(var.asInt() - 1);
      return Ok<Value, ExecutionError>(oldValue);
    }
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  default:
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
}

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

Result<Value, ExecutionError>
LaneIndexExpr::evaluate_result(LaneContext &lane, WaveContext &,
                               ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(lane.laneId)));
}

Result<Value, ExecutionError>
WaveIndexExpr::evaluate_result(LaneContext &, WaveContext &wave,
                               ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(wave.waveId)));
}

Result<Value, ExecutionError>
ThreadIndexExpr::evaluate_result(LaneContext &lane, WaveContext &,
                                 ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(lane.laneId)));
}

Result<Value, ExecutionError>
ConditionalExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                 ThreadgroupContext &tg) const {
#ifdef _MSC_VER
  auto _result = condition_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  auto condResult = _result.unwrap();
#else
  auto condResult = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif

  if (condResult.asInt()) {
    return trueExpr_->evaluate_result(lane, wave, tg);
  } else {
    return falseExpr_->evaluate_result(lane, wave, tg);
  }
}

Result<Value, ExecutionError>
WaveGetLaneCountExpr::evaluate_result(LaneContext &, WaveContext &wave,
                                      ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(wave.waveSize)));
}

Result<Value, ExecutionError>
WaveIsFirstLaneExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);

  // Find the first active lane in the wave
  for (LaneId lid = 0; lid < wave.lanes.size(); ++lid) {
    if (wave.lanes[lid]->isActive) {
      return Ok<Value, ExecutionError>(Value(lane.laneId == lid));
    }
  }

  return Err<Value, ExecutionError>(ExecutionError::InvalidState);
}

Result<Value, ExecutionError>
WaveReadLaneAt::evaluate_result(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  
  // Evaluate the lane index
#ifdef _MSC_VER
  auto _result = laneIndex_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  auto laneIndexResult = _result.unwrap();
#else
  auto laneIndexResult = TRY_RESULT(laneIndex_->evaluate_result(lane, wave, tg), Value, ExecutionError);
#endif
  
  LaneId targetLane = laneIndexResult.asInt();
  
  // Check if target lane is valid
  if (targetLane >= wave.lanes.size())
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  
  // Check if target lane is active
  if (!wave.lanes[targetLane]->isActive)
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  
  // Evaluate the value expression in the context of the target lane
  // This is the key: we evaluate the expression using the target lane's context
  return value_->evaluate_result(*wave.lanes[targetLane], wave, tg);
}

Result<Value, ExecutionError>
SharedReadExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                ThreadgroupContext &tg) const {
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  return Ok<Value, ExecutionError>(tg.sharedMemory->read(addr_, tid));
}

Result<Value, ExecutionError>
BufferAccessExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                  ThreadgroupContext &tg) const {
#ifdef _MSC_VER
  auto _result = indexExpr_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  Value indexValue = _result.unwrap();
#else
  Value indexValue = TRY_RESULT(indexExpr_->evaluate_result(lane, wave, tg),
                                Value, ExecutionError);
#endif
  uint32_t index = indexValue.asInt();

  // Look up the global buffer
  auto bufferIt = tg.globalBuffers.find(bufferName_);
  if (bufferIt == tg.globalBuffers.end()) {
    Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }

  return Ok<Value, ExecutionError>(bufferIt->second->load(index));
}

Result<Value, ExecutionError>
ArrayAccessExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                ThreadgroupContext &tg) const {
  // Evaluate index
#ifdef _MSC_VER
  auto _result = indexExpr_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
  Value indexValue = _result.unwrap();
#else
  Value indexValue = TRY_RESULT(indexExpr_->evaluate_result(lane, wave, tg),
                                Value, ExecutionError);
#endif
  uint32_t index = indexValue.asInt();
  
  // Check if this is a global buffer first
  if (tg.globalBuffers.count(arrayName_)) {
    auto& buffer = tg.globalBuffers.at(arrayName_);
    Value result = buffer->load(index);
    INTERPRETER_DEBUG_LOG("Loaded value " << result.toString() << " from global buffer " 
              << arrayName_ << "[" << index << "]");
    return Ok<Value, ExecutionError>(result);
  }
  
  // Otherwise, look for indexed variable
  std::string indexedName = arrayName_ + "_" + std::to_string(index);
  auto it = lane.variables.find(indexedName);
  if (it != lane.variables.end()) {
    return Ok<Value, ExecutionError>(it->second);
  }
  
  // Finally, check if it's a non-indexed array variable
  auto it2 = lane.variables.find(arrayName_);
  if (it2 != lane.variables.end()) {
    return Ok<Value, ExecutionError>(it2->second);
  }
  
  return Err<Value, ExecutionError>(ExecutionError::InvalidState);
}

Result<Value, ExecutionError>
DispatchThreadIdExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg) const {
  // Calculate global thread ID based on wave and lane
  // For 1D dispatch: tid = waveId * waveSize + laneId
  uint32_t globalTid = wave.waveId * wave.waveSize + lane.laneId;
  
  // Handle different components
  switch (component_) {
    case 0: // x component
      return Ok<Value, ExecutionError>(Value(static_cast<int>(globalTid)));
    case 1: // y component
      return Ok<Value, ExecutionError>(Value(0)); // Assuming 1D dispatch
    case 2: // z component
      return Ok<Value, ExecutionError>(Value(0)); // Assuming 1D dispatch
    default:
      return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
}

// Wave operation implementations
WaveActiveOp::WaveActiveOp(std::unique_ptr<Expression> expr, OpType op)
    : Expression(""), expr_(std::move(expr)), op_(op) {}

Result<Value, ExecutionError>
WaveActiveOp::evaluate_result(LaneContext &lane, WaveContext &wave,
                              ThreadgroupContext &tg) const {
  // Wave operations require all active lanes to participate
  if (!lane.isActive) {
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }

  // Create block-scoped instruction identity using compound key
  uint32_t currentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  std::pair<const void *, uint32_t> instructionKey = {
      static_cast<const void *>(this), currentBlockId};

  INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Lane " << lane.laneId
            << " executing WaveActiveSum in block " << currentBlockId
            << ", instruction key=(" << static_cast<const void *>(this) << ","
            << currentBlockId << ")");

  // CRITICAL: If lane is resuming from wave operation, check for stored results
  // first
  if (lane.isResumingFromWaveOp) {
    INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Lane " << lane.laneId
              << " is resuming from wave operation, checking for stored result");

    auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
    if (syncPointIt != wave.activeSyncPoints.end()) {
      auto &syncPoint = syncPointIt->second;
      if (syncPoint.getPhase() == SyncPointState::Executed) {
        // Use state machine method to retrieve result
#ifdef _MSC_VER
        auto _result = syncPoint.retrieveResult(lane.laneId);
        if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
        Value result = _result.unwrap();
#else
        Value result = TRY_RESULT(syncPoint.retrieveResult(lane.laneId), Value,
                                  ExecutionError);
#endif
        INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Lane " << lane.laneId
                  << " retrieving stored wave result: " << result.toString()
                  << " (phase: " << syncPointStateToString(syncPoint.getPhase())
                  << ")");

        // Clear the resuming flag - we successfully retrieved the result
        const_cast<LaneContext &>(lane).isResumingFromWaveOp = false;
        return Ok<Value, ExecutionError>(std::move(result));
      }
    }

    // No stored result found - clear flag and continue with normal execution
    INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Lane " << lane.laneId
              << " no stored result found for key (" << instructionKey.first
              << "," << instructionKey.second
              << "), continuing with normal execution");

    const_cast<LaneContext &>(lane).isResumingFromWaveOp = false;
  }

  // Check if there's already a computed result for this lane (normal path)
  auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
  if (syncPointIt != wave.activeSyncPoints.end()) {
    auto &syncPoint = syncPointIt->second;
    if (syncPoint.getPhase() == SyncPointState::Executed) {
      // Use state machine method to retrieve result
#ifdef _MSC_VER
      auto _result = syncPoint.retrieveResult(lane.laneId);
      if (_result.is_err()) return Err<Value, ExecutionError>(_result.unwrap_err());
      Value result = _result.unwrap();
#else
      Value result = TRY_RESULT(syncPoint.retrieveResult(lane.laneId), Value,
                                ExecutionError);
#endif
      INTERPRETER_DEBUG_LOG(
          "Lane " << lane.laneId
                  << " retrieving stored wave result: " << result.toString()
                  << " (phase: " << (int)syncPoint.getPhase() << ")\n");

      return Ok<Value, ExecutionError>(std::move(result));
    }
  }

  // Mark this lane as waiting at this specific instruction
  tg.markLaneWaitingAtWaveInstruction(wave.waveId, lane.laneId,
                                      static_cast<const void *>(this),
                                      toString(),
                                      static_cast<int>(op_));

  // No stored result - need to wait
  if (!lane.isResumingFromWaveOp) {
    // Store the compound key in the sync point for proper tracking
    auto &syncPoint = wave.activeSyncPoints[instructionKey];
    syncPoint.instruction = static_cast<const void *>(this);

    // Get current block and mark lane as waiting
    tg.markLaneWaitingForWave(wave.waveId, lane.laneId, currentBlockId);

    INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Lane " << lane.laneId
              << " cannot execute, starting to wait in block " << currentBlockId);

    // Check if this newly waiting lane completes the participant set
    if (tg.canExecuteWaveInstruction(wave.waveId, lane.laneId,
                                     static_cast<const void *>(this))) {
      INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: After lane " << lane.laneId
                << " started waiting, wave operation can now execute!");
    }

    // Return error to indicate we need to wait
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }

  // This shouldn't happen in the new collective model
  return Err<Value, ExecutionError>(ExecutionError::InvalidState);
}

Value WaveActiveOp::computeWaveOperation(
    const std::vector<Value> &values) const {
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
    const Value &firstValue = values[0];
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
    // In real HLSL, this would return a 32-bit or 64-bit mask depending on wave
    // size
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
  static const char *opNames[] = {
      "WaveActiveSum",     "WaveActiveProduct",   "WaveActiveMin",
      "WaveActiveMax",     "WaveActiveBitAnd",    "WaveActiveBitOr",
      "WaveActiveBitXor",  "WaveActiveCountBits", "WaveActiveAllTrue",
      "WaveActiveAnyTrue", "WaveActiveAllEqual",  "WaveActiveBallot"};
  return std::string(opNames[op_]) + "(" + expr_->toString() + ")";
}

// Forward declaration removed - no longer needed

// Statement implementations
// Legacy constructor - tries to get type from init expression
VarDeclStmt::VarDeclStmt(const std::string &name,
                         std::unique_ptr<Expression> init)
    : name_(name), type_(init ? init->getType() : HLSLType::Uint), init_(std::move(init)) {}

// Constructor with explicit type
VarDeclStmt::VarDeclStmt(const std::string &name, HLSLType type,
                         std::unique_ptr<Expression> init)
    : name_(name), type_(type), init_(std::move(init)) {}

// Constructor with string type (for compatibility)
VarDeclStmt::VarDeclStmt(const std::string &name, const std::string &typeStr,
                         std::unique_ptr<Expression> init)
    : name_(name), type_(HLSLTypeInfo::fromString(typeStr)), init_(std::move(init)) {
  if (type_ == HLSLType::Custom) {
    customTypeStr_ = typeStr;
  }
}

Result<Unit, ExecutionError>
VarDeclStmt::execute_result(LaneContext &lane, WaveContext &wave,
                            ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based implementation - no exceptions!
  Value initVal;
  if (init_) {
#ifdef _MSC_VER
    auto _result = init_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    initVal = _result.unwrap();
#else
    initVal = TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit,
                         ExecutionError);
#endif
  } else {
    initVal = Value(0);
  }

  lane.variables[name_] = initVal;
  
  // Debug output for variable assignment
  INTERPRETER_DEBUG_LOG("DEBUG: VarDecl - Lane " << lane.laneId << " assigned " 
            << name_ << " = " << initVal.toString());
  
  // Call hook for variable write access
  if (tg.interpreter) {
    tg.interpreter->onVariableAccess(lane, wave, tg, name_, true, initVal);
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string VarDeclStmt::toString() const {
  std::string typeStr = type_ == HLSLType::Custom ? customTypeStr_ : HLSLTypeInfo::toString(type_);
  return typeStr + " " + name_ + " = " + (init_ ? init_->toString() : "0") + ";";
}

AssignStmt::AssignStmt(const std::string &name,
                       std::unique_ptr<Expression> expr)
    : name_(name), expr_(std::move(expr)) {}

Result<Unit, ExecutionError>
AssignStmt::execute_result(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based implementation - no exceptions!
#ifdef _MSC_VER
  auto _result = expr_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
  Value val = _result.unwrap();
#else
  Value val =
      TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
#endif

  INTERPRETER_DEBUG_LOG("DEBUG: AssignStmt - Lane " << lane.laneId << " assigned value "
            << val.toString() << " to variable '" << name_ << "'");

  lane.variables[name_] = val;
  
  // Call hook for variable write access
  if (tg.interpreter) {
    tg.interpreter->onVariableAccess(lane, wave, tg, name_, true, val);
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string AssignStmt::toString() const {
  return name_ + " = " + expr_->toString() + ";";
}

Result<Unit, ExecutionError>
ArrayAssignStmt::execute_result(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});
  
  // Evaluate index
#ifdef _MSC_VER
  auto _result = indexExpr_->evaluate_result(lane, wave, tg);
  if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
  Value indexValue = _result.unwrap();
#else
  Value indexValue = TRY_RESULT(indexExpr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
#endif
  uint32_t index = indexValue.asInt();
  
  // Evaluate value to assign
#ifdef _MSC_VER
  auto _result2 = valueExpr_->evaluate_result(lane, wave, tg);
  if (_result2.is_err()) return Err<Unit, ExecutionError>(_result2.unwrap_err());
  Value val = _result2.unwrap();
#else
  Value val = TRY_RESULT(valueExpr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
#endif
  
  // Check if this is a global buffer
  if (tg.globalBuffers.count(arrayName_)) {
    auto& buffer = tg.globalBuffers[arrayName_];
    buffer->store(index, val);
  } else {
    // Fall back to local variable with indexed name
    std::string indexedName = arrayName_ + "_" + std::to_string(index);
    lane.variables[indexedName] = val;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string ArrayAssignStmt::toString() const {
  return arrayName_ + "[" + indexExpr_->toString() + "] = " + valueExpr_->toString() + ";";
}

IfStmt::IfStmt(std::unique_ptr<Expression> cond,
               std::vector<std::unique_ptr<Statement>> thenBlock,
               std::vector<std::unique_ptr<Statement>> elseBlock)
    : condition_(std::move(cond)), thenBlock_(std::move(thenBlock)),
      elseBlock_(std::move(elseBlock)) {}

void IfStmt::performReconvergence(LaneContext &lane, WaveContext &wave,
                                  ThreadgroupContext &tg, int ourStackIndex,
                                  bool hasElse) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  uint32_t currentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  uint32_t laneSpecificMergeBlockId = ourEntry.ifMergeBlockId;
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " performing reconvergence from block " << currentBlockId
            << " to laneSpecificMergeBlockId=" << laneSpecificMergeBlockId);

  // Debug: Show current merge stack before popping
  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " merge stack before reconvergence (size="
            << currentMergeStack.size() << "):");
  for (size_t i = 0; i < currentMergeStack.size(); i++) {
    INTERPRETER_DEBUG_LOG("  Stack[" << i
              << "]: sourceStatement=" << currentMergeStack[i].sourceStatement);
  }

  // Use stored merge block ID - don't recreate blocks during reconvergence
  // mergeBlockId should already be set from initial setup

  // Clean up execution state
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " popping stack at reconvergence (depth "
            << lane.executionStack.size() << "->"
            << (lane.executionStack.size() - 1) << ", this=" << this << ")");
  lane.executionStack.pop_back();

  // Reconverge at merge block
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " popping merge point before assigning to block "
            << laneSpecificMergeBlockId);
  tg.popMergePoint(wave.waveId, lane.laneId);
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " assigning to merge block " << laneSpecificMergeBlockId);
  // tg.assignLaneToBlock(wave.waveId, lane.laneId,
  // laneSpecificMergeBlockId);

  // Move lane to merge block as participating (reconvergence)
  tg.moveThreadFromUnknownToParticipating(laneSpecificMergeBlockId, wave.waveId,
                                          lane.laneId);

  // Clean up then/else blocks - lane will never return to them
  bool setupComplete =
      (ourEntry.ifThenBlockId != 0 || ourEntry.ifElseBlockId != 0 ||
       ourEntry.ifMergeBlockId != 0);
  if (setupComplete) {
    tg.removeThreadFromAllSets(ourEntry.ifThenBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(ourEntry.ifThenBlockId, wave.waveId,
                                    lane.laneId);

    if (hasElse && ourEntry.ifElseBlockId != 0) {
      tg.removeThreadFromAllSets(ourEntry.ifElseBlockId, wave.waveId,
                                 lane.laneId);
      tg.removeThreadFromNestedBlocks(ourEntry.ifElseBlockId, wave.waveId,
                                      lane.laneId);
    }
  }

  // Restore active state (reconvergence)
  lane.isActive = lane.isActive && !lane.hasReturned;

  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " reconvergence complete");
}

Result<Unit, ExecutionError> IfStmt::execute_result(LaneContext &lane,
                                                    WaveContext &wave,
                                                    ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);

  bool isResuming = (ourStackIndex >= 0);

  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting")
            << " Result-based if statement");

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " starting fresh execution (pushed to stack depth="
              << lane.executionStack.size() << ", this=" << this << ")");
  }

  // Don't hold reference to vector element - it can be invalidated during
  // nested execution
  bool hasElse = !elseBlock_.empty();
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Result-based state machine for if statement execution
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId << " in phase "
            << LaneContext::getPhaseString(ourEntry.phase)
            << " (Result-based, stack depth=" << lane.executionStack.size()
            << ", our index=" << ourStackIndex << ", this=" << this << ")");

  switch (ourEntry.phase) {
  case LaneContext::ControlFlowPhase::EvaluatingCondition: {
    // Evaluate condition and setup blocks using Result-based helper method
#ifdef _MSC_VER
    auto _result = evaluateConditionAndSetup_result(lane, wave, tg, ourStackIndex,
                                                parentBlockId, hasElse);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
#else
    TRY_RESULT(evaluateConditionAndSetup_result(lane, wave, tg, ourStackIndex,
                                                parentBlockId, hasElse),
               Unit, ExecutionError);
#endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::ExecutingBody: {
    // For IfStmt, ExecutingBody phase handles both then and else branches
    // Check which branch to execute based on saved condition result
    if (ourEntry.inThenBranch) {
      // Execute then branch using Result-based helper method
#ifdef _MSC_VER
      auto _result = executeThenBranch_result(lane, wave, tg, ourStackIndex);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
#else
      TRY_RESULT(executeThenBranch_result(lane, wave, tg, ourStackIndex), Unit,
                 ExecutionError);
#endif
    } else {
      // Execute else branch using Result-based helper method
#ifdef _MSC_VER
      auto _result = executeElseBranch_result(lane, wave, tg, ourStackIndex);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
#else
      TRY_RESULT(executeElseBranch_result(lane, wave, tg, ourStackIndex), Unit,
                 ExecutionError);
#endif
    }

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::Reconverging: {
    // Perform reconvergence using non-throwing helper method
    performReconvergence(lane, wave, tg, ourStackIndex, hasElse);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  default:
    INTERPRETER_DEBUG_LOG("ERROR: IfStmt - Lane " << lane.laneId
              << " unexpected phase in Result-based execution");
    lane.executionStack.pop_back();
    return Err<Unit, ExecutionError>(ExecutionError::InvalidState);
  }
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

// Specialized wrapper function for IfStmt-specific error handling
Result<Unit, ExecutionError>
IfStmt::execute_with_error_handling(LaneContext &lane, WaveContext &wave,
                                    ThreadgroupContext &tg) {
  auto result = execute_result(lane, wave, tg);
  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();

    switch (error) {
    case ExecutionError::WaveOperationWait:
      // IfStmt-specific: Wave operation already set the lane to WaitingForWave
      // Just propagate the error without changing state
      return result; // Propagate for parent to handle

    case ExecutionError::ControlFlowBreak:
    case ExecutionError::ControlFlowContinue:
      // IfStmt does NOT consume break/continue - propagate to enclosing loop
      // Clean up our stack entry first
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          // Get block IDs before popping the stack
          uint32_t ifThenBlockId =
              lane.executionStack[ourStackIndex].ifThenBlockId;
          uint32_t ifElseBlockId =
              lane.executionStack[ourStackIndex].ifElseBlockId;
          uint32_t ifMergeBlockId =
              lane.executionStack[ourStackIndex].ifMergeBlockId;

          INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
                    << " cleaning up for continue (popping stack from depth "
                    << lane.executionStack.size() << " to "
                    << (lane.executionStack.size() - 1) << ", this=" << this
                    << ")");

          lane.executionStack.pop_back();
          tg.popMergePoint(wave.waveId, lane.laneId);

          // Clean up then/else blocks - lane will never return to them
          tg.removeThreadFromAllSets(ifThenBlockId, wave.waveId, lane.laneId);
          tg.removeThreadFromNestedBlocks(ifThenBlockId, wave.waveId,
                                          lane.laneId);
          if (ifElseBlockId != 0) {
            tg.removeThreadFromAllSets(ifElseBlockId, wave.waveId, lane.laneId);
            tg.removeThreadFromNestedBlocks(ifElseBlockId, wave.waveId,
                                            lane.laneId);
          }

          // Also clean up merge block since we're not going there
          tg.removeThreadFromAllSets(ifMergeBlockId, wave.waveId, lane.laneId);
          tg.removeThreadFromNestedBlocks(ifMergeBlockId, wave.waveId,
                                          lane.laneId);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return result; // Propagate to parent loop

    default:
      return Err<Unit, ExecutionError>(ExecutionError::InvalidState);
    }
  }
  return result; // Success case
}

// Result-based versions of IfStmt helper methods
Result<Unit, ExecutionError> IfStmt::evaluateConditionAndSetup_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t parentBlockId, bool hasElse) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)");

  // Only evaluate condition if not already evaluated (avoid re-evaluation on
  // resume)
  if (!ourEntry.conditionEvaluated) {
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " evaluating condition for first time (Result-based)");

    // Evaluate condition using Result-based evaluation
#ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
#else
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg),
                               Unit, ExecutionError);
#endif
    bool conditionResult = condVal.asBool();

    lane.executionStack[ourStackIndex].conditionResult = conditionResult;
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " condition result=" << ourEntry.conditionResult
              << " (Result-based)");
    
    // Call control flow hook
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, conditionResult);
    }
  } else {
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " using cached condition result=" << ourEntry.conditionResult
              << " (Result-based)");
  }

  // Condition evaluated successfully - set up blocks
  std::set<uint32_t> divergentBlocks;
  tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                    parentBlockId, divergentBlocks);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  auto blockIds =
      tg.createIfBlocks(static_cast<const void *>(this), parentBlockId,
                        currentMergeStack, hasElse, lane.executionPath);
  ourEntry.ifThenBlockId = std::get<0>(blockIds);
  ourEntry.ifElseBlockId = std::get<1>(blockIds);
  ourEntry.ifMergeBlockId = std::get<2>(blockIds);
  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " setup complete: thenBlockId=" << ourEntry.ifThenBlockId
            << ", elseBlockId=" << ourEntry.ifElseBlockId
            << ", mergeBlockId=" << ourEntry.ifMergeBlockId << " (Result-based)");

  // Update blocks based on condition result
  if (ourEntry.conditionResult) {
    tg.moveThreadFromUnknownToParticipating(ourEntry.ifThenBlockId, wave.waveId,
                                            lane.laneId);
    if (hasElse) {
      tg.removeThreadFromUnknown(ourEntry.ifElseBlockId, wave.waveId,
                                 lane.laneId);
      tg.removeThreadFromNestedBlocks(ourEntry.ifElseBlockId, wave.waveId,
                                      lane.laneId);
    }
    // Don't remove from merge block yet - lane will reconverge there later
    ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
    ourEntry.inThenBranch = true;
    ourEntry.statementIndex = 0;
  } else {
    tg.removeThreadFromUnknown(ourEntry.ifThenBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(ourEntry.ifThenBlockId, wave.waveId,
                                    lane.laneId);
    if (hasElse) {
      tg.moveThreadFromUnknownToParticipating(ourEntry.ifElseBlockId,
                                              wave.waveId, lane.laneId);
      // Don't remove from merge block yet - lane will reconverge there later
      ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
      ourEntry.inThenBranch = false;
      ourEntry.statementIndex = 0;
    } else {
      // No else block - go directly to merge
      ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    }
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
IfStmt::executeThenBranch_result(LaneContext &lane, WaveContext &wave,
                                 ThreadgroupContext &tg, int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " executing then block from statement "
            << ourEntry.statementIndex << " of " << thenBlock_.size()
            << " total statements (Result-based)");

  // Execute statements in then block from saved position
  for (size_t i = ourEntry.statementIndex; i < thenBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " executing then block statement " << i << ": "
              << thenBlock_[i]->toString());

    // Debug: Print variables before statement
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " variables BEFORE statement: ");
    for (const auto &var : lane.variables) {
      INTERPRETER_DEBUG_LOG(var.first << "=" << var.second.toString() << " ");
    }

    // Use Result-based execute_with_error_handling for proper control flow
    // handling
    auto stmt_result =
        thenBlock_[i]->execute_with_error_handling(lane, wave, tg);
    if (stmt_result.is_err()) {
      return stmt_result; // Propagate the error
    }

    // Debug: Print variables after statement
    INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
              << " variables AFTER statement: ");
    for (const auto &var : lane.variables) {
      INTERPRETER_DEBUG_LOG(var.first << "=" << var.second.toString() << " ");
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.hasReturned) {
      INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ") (Result-based)");
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  // All statements executed - transition to reconverging
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
IfStmt::executeElseBranch_result(LaneContext &lane, WaveContext &wave,
                                 ThreadgroupContext &tg, int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
            << " executing else block from statement "
            << ourEntry.statementIndex << " (Result-based)");

  // Execute statements in else block from saved position
  for (size_t i = ourEntry.statementIndex; i < elseBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    // Use Result-based execute_with_error_handling for proper control flow
    // handling
    auto stmt_result =
        elseBlock_[i]->execute_with_error_handling(lane, wave, tg);
    if (stmt_result.is_err()) {
      return stmt_result; // Propagate the error
    }

    if (lane.hasReturned) {
      INTERPRETER_DEBUG_LOG("DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ") (Result-based)");
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return Ok<Unit, ExecutionError>(Unit{});
    }

    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  // All statements executed - transition to reconverging
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;

  return Ok<Unit, ExecutionError>(Unit{});
}

ForStmt::ForStmt(const std::string &var, std::unique_ptr<Expression> init,
                 std::unique_ptr<Expression> cond,
                 std::unique_ptr<Expression> inc,
                 std::vector<std::unique_ptr<Statement>> body)
    : loopVar_(var), init_(std::move(init)), condition_(std::move(cond)),
      increment_(std::move(inc)), body_(std::move(body)) {}

// Pure Result-based ForStmt phase implementations
Result<Unit, ExecutionError> ForStmt::executeInit(LaneContext &lane,
                                                  WaveContext &wave,
                                                  ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " executing init (Result-based)");

  // Initialize loop variable using Result-based evaluation
  #ifdef _MSC_VER
    auto _result = init_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value initVal = _result.unwrap();
  #else
  Value initVal =
      TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  #endif
  lane.variables[loopVar_] = initVal;

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<bool, ExecutionError>
ForStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)");

  // Evaluate condition using Result-based evaluation
  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<bool, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), bool,
                             ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError> ForStmt::executeBody(LaneContext &lane,
                                                  WaveContext &wave,
                                                  ThreadgroupContext &tg,
                                                  size_t &statementIndex) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " executing body (Result-based)");

  // Execute body statements using Result-based approach
  for (size_t i = statementIndex; i < body_.size(); ++i) {
    auto result = body_[i]->execute_result(lane, wave, tg);
    if (result.is_err()) {
      // Handle control flow errors
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        INTERPRETER_DEBUG_LOG("ForStmt - Break encountered in body");
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Continue encountered in body");
        statementIndex = body_.size();           // Skip remaining statements
        return Ok<Unit, ExecutionError>(Unit{}); // Continue to increment phase
      } else {
        // Other errors (like WaveOperationWait) should be propagated
        statementIndex = i; // Save position for resumption
        return result;
      }
    }
  }

  statementIndex = 0; // Reset for next iteration
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::executeIncrement(LaneContext &lane,
                                                       WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " executing increment (Result-based)");

    // Execute increment expression using Result-based evaluation
  #ifdef _MSC_VER
    auto _result = increment_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
  #else
    TRY_RESULT(increment_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  #endif

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::execute_result(LaneContext &lane,
                                                     WaveContext &wave,
                                                     ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);

  bool isResuming = (ourStackIndex >= 0);
  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;

  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting")
            << " Result-based for loop execution");

  if (!isResuming) {
    // Starting fresh - push initial state for initialization
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingInit);
    ourStackIndex = lane.executionStack.size() - 1;

    // Set up fresh execution using non-throwing helper method
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId,
                        mergeBlockId);
  } else {
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
              << " resuming execution (found at stack index=" << ourStackIndex
              << ", current stack depth=" << lane.executionStack.size()
              << ", this=" << this << ")");

    // Restore saved block IDs
    headerBlockId = lane.executionStack[ourStackIndex].loopHeaderBlockId;
    mergeBlockId = lane.executionStack[ourStackIndex].loopMergeBlockId;
  }

  // Execute loop with Result-based state machine
  auto &ourEntry = lane.executionStack[ourStackIndex];
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId << " in phase "
            << LaneContext::getPhaseString(ourEntry.phase)
            << " (Result-based, stack depth=" << lane.executionStack.size()
            << ", our index=" << ourStackIndex << ", this=" << this << ")");

  switch (ourEntry.phase) {
  case LaneContext::ControlFlowPhase::EvaluatingInit: {
    // Evaluate initialization using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateInitPhase_result(lane, wave, tg, ourStackIndex, headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(
        evaluateInitPhase_result(lane, wave, tg, ourStackIndex, headerBlockId),
        Unit, ExecutionError);   
    #endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::EvaluatingCondition: {
    // Evaluate condition using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
      TRY_RESULT(evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId),
               Unit, ExecutionError);
    #endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::ExecutingBody: {
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
              << " executing body for iteration " << ourEntry.loopIteration
              << " from statement " << ourEntry.statementIndex
              << " (Result-based)");

    // Only set up iteration blocks if we're starting the body (statement index 0)
    if (ourEntry.statementIndex == 0) {
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);
    }

    // Execute body statements using Result-based helper method
    #ifdef _MSC_VER
      auto _result = executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
      TRY_RESULT(executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId),
               Unit, ExecutionError);
    #endif

    // Check if we need to return early (lane returned or needs resume)
    if (lane.hasReturned || lane.state != ThreadState::Ready) {
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Check if body is complete (all statements executed)
    if (ourEntry.statementIndex >= body_.size()) {
      // Clean up after body execution using non-throwing helper method
      cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);

      // Set state to WaitingForResume to prevent currentStatement increment
      setThreadStateIfUnprotected(lane);
    }
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::EvaluatingIncrement: {
    // Evaluate increment using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateIncrementPhase_result(lane, wave, tg, ourStackIndex);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(evaluateIncrementPhase_result(lane, wave, tg, ourStackIndex),
               Unit, ExecutionError);
    #endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::Reconverging: {
    // Handle loop exit using non-throwing helper method
    handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  default:
    INTERPRETER_DEBUG_LOG("ERROR: ForStmt - Lane " << lane.laneId
              << " unexpected phase in Result-based execution");
    lane.executionStack.pop_back();
    return Ok<Unit, ExecutionError>(Unit{});
  }
}

// Helper method for setting up iteration-specific blocks in ForStmt
void ForStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave,
                                   ThreadgroupContext &tg, int ourStackIndex,
                                   uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Check if we need an iteration-specific starting block
  // Only create one if we have non-control-flow statements before any control
  // flow
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique
  // iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      // We have regular statements that need unique iteration context
      needsIterationBlock = true;
      INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
          << " needs iteration block (first statement is not control flow)");
    } else {
INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
          << " no iteration block needed (first statement is control flow)");
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block (unique context for this
    // iteration)
    iterationStartBlockId = ourEntry.loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      // Create unique identity for this iteration's starting block
      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x3000);

      // Use REGULAR block type - this is the starting context for this
      // iteration
      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId, currentMergeStack,
          true, lane.executionPath);

      // Try to find existing block first
      iterationStartBlockId = tg.findBlockByIdentity(iterationIdentity);

      if (iterationStartBlockId == 0) {
        // Create new iteration starting block
        std::map<WaveId, std::set<LaneId>> expectedLanes =
            tg.getCurrentBlockParticipants(headerBlockId);
        iterationStartBlockId =
            tg.findOrCreateBlockForPath(iterationIdentity, expectedLanes);
INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                  << " created iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration);
      } else {
        INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                  << " found existing iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration);
      }

      ourEntry.loopBodyBlockId = iterationStartBlockId;
    }

    // Move to iteration-specific block
    if (tg.getCurrentBlock(wave.waveId, lane.laneId) != iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId,
                                              wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                << " moved to iteration starting block "
                << iterationStartBlockId);
    }
  } else {
    // No iteration block needed - but we still need to ensure unique execution
    // context per iteration Only push merge point if we're at the beginning of
    // the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees
      // different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                << " merge stack size: " << currentMergeStack.size());
      for (size_t i = 0; i < currentMergeStack.size(); i++) {
        INTERPRETER_DEBUG_LOG("  Stack[" << i << "]: sourceStatement="
                  << currentMergeStack[i].sourceStatement);
      }
      INTERPRETER_DEBUG_LOG("  Looking for iterationMarker=" << iterationMarker);
      bool alreadyPushed =
          hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack");
      }

      if (!alreadyPushed) {
        std::set<uint32_t>
            emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId),
                          emptyDivergentBlocks);
        INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                  << " pushed iteration merge point " << iterationMarker
                  << " (no iteration block needed, but merge stack modified)");
      }
    }
  }
}

// Helper method for body completion cleanup in ForStmt
void ForStmt::cleanupAfterBodyExecution(LaneContext &lane, WaveContext &wave,
                                        ThreadgroupContext &tg,
                                        int ourStackIndex,
                                        uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Body completed - clean up iteration-specific merge point
  const void *iterationMarker =
      reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(this) +
                                     (ourEntry.loopIteration << 16) + 0x5000);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration " << ourEntry.loopIteration);
  }

  // Move back to header block for increment phase
  uint32_t finalBlock =
      tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " body completed in block " << finalBlock
            << ", moving to header block " << headerBlockId << " for increment");
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  ourEntry.phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;
  setThreadStateIfUnprotected(lane);
}

// Helper method for loop exit/reconverging phase in ForStmt
void ForStmt::handleLoopExit(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg, int ourStackIndex,
                             uint32_t mergeBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId << " exiting loop after "
            << ourEntry.loopIteration << " iterations"
            << " (state before=" << (int)lane.state << ")");

  // Clean up execution state
  lane.executionStack.pop_back();

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId,
                                          lane.laneId);

  // Loop has completed, restore active state and reset thread state
  lane.isActive = lane.isActive && !lane.hasReturned;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::Ready;
  }
}

// Helper method for break exception handling in ForStmt
void ForStmt::handleBreakException(LaneContext &lane, WaveContext &wave,
                                   ThreadgroupContext &tg, int ourStackIndex,
                                   uint32_t headerBlockId) {
  // Break - exit loop
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId << " breaking from loop");
  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
  setThreadStateIfUnprotected(lane);
}

// Helper method for continue exception handling in ForStmt
void ForStmt::handleContinueException(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg, int ourStackIndex,
                                      uint32_t headerBlockId) {
  // Continue - go to increment phase and skip remaining statements
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId << " continuing loop");

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId,
                                        ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Marked lane " << lane.laneId
              << " as Left from block " << ourEntry.loopBodyBlockId);
  }

  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from all nested blocks this lane is abandoning
  if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
    tg.removeThreadFromAllSets(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
    tg.removeThreadFromNestedBlocks(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
  }
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  lane.executionStack[ourStackIndex].loopBodyBlockId =
      0; // Reset for next iteration
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingIncrement;

  // Set state to WaitingForResume to prevent currentStatement increment
  setThreadStateIfUnprotected(lane);
}

// Helper method for fresh execution setup in ForStmt
void ForStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave,
                                  ThreadgroupContext &tg, int ourStackIndex,
                                  uint32_t &headerBlockId,
                                  uint32_t &mergeBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " starting fresh execution (pushed to stack depth="
            << lane.executionStack.size() << ", this=" << this << ")");

  // Get current block before entering loop
  uint32_t parentBlockId =
      tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);

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
}

// Result-based versions of ForStmt helper methods
Result<Unit, ExecutionError>
ForStmt::executeBodyStatements_result(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg, int ourStackIndex,
                                      uint32_t headerBlockId) {
  // Execute statements - start in iteration block, naturally flow to merge
  // blocks
  for (size_t i = lane.executionStack[ourStackIndex].statementIndex;
       i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement =
        tg.getCurrentBlock(wave.waveId, lane.laneId);
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
              << " executing statement " << i << " in block "
              << blockBeforeStatement << " (Result-based)");

    // Use Result-based execute_with_error_handling for proper control flow
    // handling
    auto stmt_result = body_[i]->execute_with_error_handling(lane, wave, tg);
    if (stmt_result.is_err()) {
      return stmt_result; // Propagate the error
    }

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (lane.executionStack[ourStackIndex].loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG(
            "DEBUG: ForStmt - Lane " << lane.laneId
            << " popped iteration merge point on early return (Result-based)");
      }
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement =
          tg.getCurrentBlock(wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                << " child statement needs resume (Result-based)");
      INTERPRETER_DEBUG_LOG("  Block before: " << blockBeforeStatement
                << ", Block after: " << blockAfterStatement
                << ", statementIndex stays at: "
                << lane.executionStack[ourStackIndex].statementIndex);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
                << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement "
                << i << " (likely merge block, Result-based)");
    }

    // Update statement index
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
              << " completed statement " << i
              << ", incrementing statementIndex to " << (i + 1));
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
ForStmt::evaluateInitPhase_result(LaneContext &lane, WaveContext &wave,
                                  ThreadgroupContext &tg, int ourStackIndex,
                                  uint32_t headerBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating init (Result-based)");
            
  #ifdef _MSC_VER
    auto _result = init_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value initVal = _result.unwrap();
  #else
  // Initialize loop variable using Result-based evaluation
  Value initVal =
      TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  #endif
  lane.variables[loopVar_] = initVal;

  // Move to loop header block
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);

  // Move to condition evaluation phase
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration = 0;
  setThreadStateIfUnprotected(lane);

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::evaluateConditionPhase_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration
            << " (Result-based)");

  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    // Check loop condition using Result-based evaluation
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit,
                             ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all
    // iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId,
                               lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(
        headerBlockId, wave.waveId,
        lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    lane.executionStack[ourStackIndex].phase =
        LaneContext::ControlFlowPhase::Reconverging;
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  // Condition passed, move to body execution
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  setThreadStateIfUnprotected(lane);

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
ForStmt::evaluateIncrementPhase_result(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg,
                                       int ourStackIndex) {
  INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating increment (Result-based)");

  // Evaluate increment expression for side effects only - DO NOT assign result
  // back to loop variable For b++, the result is the old value, but the side
  // effect is incrementing b For ++b, the result is the new value, but we still
  // don't want to assign it back
  #ifdef _MSC_VER
    auto _result = increment_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
  #else
    TRY_RESULT(increment_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  #endif
  
  // Move back to condition evaluation
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration++;
  setThreadStateIfUnprotected(lane);

  return Ok<Unit, ExecutionError>(Unit{});
}

// ForStmt specialized wrapper function
Result<Unit, ExecutionError>
ForStmt::execute_with_error_handling(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg) {
  auto result = execute_result(lane, wave, tg);
  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();
    INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Handling error in wrapper: "
              << static_cast<int>(error));

    switch (error) {
    case ExecutionError::WaveOperationWait:
      // ForStmt-specific: Wave operation already set the lane to WaitingForWave
      // Just propagate the error without changing state
      return result; // Propagate for parent to handle

    case ExecutionError::ControlFlowBreak:
      // ForStmt consumes break: exit loop cleanly
      {
        INTERPRETER_DEBUG_LOG("DEBUG: ForStmt - Break consumed, exiting loop cleanly");
        // Find our stack entry to get header block ID
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId);

          // Set state to WaitingForResume
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - break handled

    case ExecutionError::ControlFlowContinue:
      // ForStmt consumes continue: jump to increment phase
      {
        INTERPRETER_DEBUG_LOG(
            "DEBUG: ForStmt - Continue consumed, jumping to increment phase");
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);

          // Set state to WaitingForResume
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - continue handled

    default:
      // Other errors propagate up unchanged
      return result;
    }
  }
  return result; // Success case
}

std::string ForStmt::toString() const {
  std::string result = "for (uint " + loopVar_ + " = " + init_->toString() + "; ";
  result += condition_->toString() + "; ";
  result += increment_->toString() + ") {\n";
  for (const auto &stmt : body_) {
    result += "    " + stmt->toString() + "\n";
  }
  result += "}";
  return result;
}

ReturnStmt::ReturnStmt(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

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
  // Lane is no longer in any block after return - registry will handle this

  // 7. Remove from lane waiting map
  wave.laneWaitingAtInstruction.erase(returningLaneId);
}

void ReturnStmt::updateBlockResolutionStates(ThreadgroupContext &tg,
                                             WaveContext &wave,
                                             LaneId returningLaneId) {
  // Remove lane from ALL execution blocks and check resolution states
  WaveId waveId = wave.waveId;
  for (auto &[blockId, block] : tg.executionBlocks) {
    // Remove lane from all block participant sets using registry as single
    // source of truth
    tg.membershipRegistry.setLaneStatus(waveId, returningLaneId, blockId,
                                        LaneBlockStatus::Left);

    // Remove from per-instruction participants in this block
    for (auto &[instruction, participants] :
         block.getInstructionParticipants()) {
      block.removeInstructionParticipant(instruction, waveId, returningLaneId);
    }

    // Check if block resolution state changed
    // Block is resolved when no unknown lanes remain (all lanes have chosen to
    // join or return)
    // bool wasResolved = block.isWaveAllUnknownResolved(waveId);

    bool registryResolved =
        tg.membershipRegistry.isWaveAllUnknownResolved(waveId, blockId);

    // Resolution status tracked by registry - old system metadata not needed

    // If block just became resolved, wake up any lanes waiting for resolution
    if (!registryResolved) {
      // All lanes in this block can now proceed with wave operations
      // Use registry instead of old block tracking
      for (LaneId laneId :
           tg.membershipRegistry.getWaitingLanes(waveId, blockId)) {
        if (laneId < wave.lanes.size() && wave.lanes[laneId] &&
            wave.lanes[laneId]->state == ThreadState::WaitingForWave) {
          // Check if this lane's wave operations can now proceed
          bool canProceed = true;
          auto waitingIt = wave.laneWaitingAtInstruction.find(laneId);
          if (waitingIt != wave.laneWaitingAtInstruction.end()) {
            const auto &instructionKey = waitingIt->second;
            canProceed = tg.canExecuteWaveInstruction(waveId, laneId,
                                                      instructionKey.first);
          }

          if (canProceed) {
            wave.lanes[laneId]->state = ThreadState::Ready;
            wave.lanes[laneId]->isResumingFromWaveOp =
                true; // Set resuming flag
          }
        }
      }
    }
  }
}

void ReturnStmt::updateWaveOperationStates(ThreadgroupContext &tg,
                                           WaveContext &wave,
                                           LaneId returningLaneId) {
  // Remove lane from ALL wave operation sync points and update completion
  // states
  INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Lane " << returningLaneId
            << " returning, checking " << wave.activeSyncPoints.size()
            << " active sync points");

  std::vector<std::pair<const void *, uint32_t>>
      completedInstructions; // Track instructions that become complete

  for (auto &[instructionKey, syncPoint] : wave.activeSyncPoints) {
    bool wasExpected =
        syncPoint.expectedParticipants.count(returningLaneId) > 0;
    INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Sync point "
              << instructionKey.first << " in block " << instructionKey.second
              << ": wasExpected=" << wasExpected << ", allParticipantsKnown="
              << syncPoint.isAllParticipantsKnown(tg, wave.waveId));

    syncPoint.expectedParticipants.erase(returningLaneId);
    syncPoint.arrivedParticipants.erase(returningLaneId);

    // If no participants left, mark this sync point for removal
    if (syncPoint.expectedParticipants.empty()) {
      INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Sync point "
                << instructionKey.first << " in block " << instructionKey.second
                << " has no participants left, marking for removal");
      completedInstructions.push_back(instructionKey);
      continue; // Skip further processing for this sync point
    }

    // Check if sync point is now ready for cleanup (has pending results)
    bool isNowReadyForCleanup = syncPoint.isReadyForCleanup();

    INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Lane " << returningLaneId
              << " returning: Block " << syncPoint.blockId
              << " isAllParticipantsKnown="
              << syncPoint.isAllParticipantsKnown(tg, wave.waveId)
              << " for instruction " << instructionKey.first);

    // If sync point is ready for cleanup, mark it for processing
    if (isNowReadyForCleanup) {
      INTERPRETER_DEBUG_LOG(
          "DEBUG: updateWaveOperationStates - Sync point for instruction "
          << instructionKey.first << " in block " << syncPoint.blockId
          << " became complete due to lane " << returningLaneId << " returning");
      completedInstructions.push_back(instructionKey);
    }
  }

  // Wake up lanes waiting at newly completed sync points
  for (const auto &instructionKey : completedInstructions) {
    auto &syncPoint = wave.activeSyncPoints[instructionKey];
    INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Waking up lanes waiting "
                 "at instruction "
              << instructionKey.first << " in block " << instructionKey.second);
    for (LaneId waitingLaneId : syncPoint.arrivedParticipants) {
      if (waitingLaneId < wave.lanes.size() && wave.lanes[waitingLaneId] &&
          wave.lanes[waitingLaneId]->state == ThreadState::WaitingForWave) {
        INTERPRETER_DEBUG_LOG("DEBUG: updateWaveOperationStates - Waking up lane "
                  << waitingLaneId << " from WaitingForWave to Ready");
        wave.lanes[waitingLaneId]->state = ThreadState::Ready;
        wave.lanes[waitingLaneId]->isResumingFromWaveOp =
            true; // Set resuming flag

        // CRITICAL FIX: Update registry status from WaitingForWave back to
        // Participating Use lane's current block, not stale block from
        // instruction key
        uint32_t currentBlockId =
            tg.getCurrentBlock(wave.waveId, waitingLaneId);
        tg.membershipRegistry.onLaneFinishWaveOp(wave.waveId, waitingLaneId,
                                                 currentBlockId);
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
      for (auto &wave : tg.waves) {
        for (auto &lane : wave->lanes) {
          if (lane->state != ThreadState::Completed &&
              lane->state != ThreadState::Error) {
            lane->state = ThreadState::Error;
            lane->errorMessage = errorMsg.str();
          }
        }
      }

      // Clear all barriers since the program is now in error state
      tg.activeBarriers.clear();
      return;
    }

    // If thread wasn't participating in this barrier, just remove it from
    // arrived list
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

ExprStmt::ExprStmt(std::unique_ptr<Expression> expr) : expr_(std::move(expr)) {}

Result<Unit, ExecutionError> ExprStmt::execute_result(LaneContext &lane,
                                                      WaveContext &wave,
                                                      ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based implementation - no exceptions!
  if (expr_) {
    // Execute the expression (evaluate it but don't store the result)
    #ifdef _MSC_VER
      auto _result = expr_->evaluate_result(lane, wave, tg);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
      TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
    #endif
  }
  return Ok<Unit, ExecutionError>(Unit{});
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

std::string SharedWriteStmt::toString() const {
  return "g_shared[" + std::to_string(addr_) + "] = " + expr_->toString() + ";";
}

// Result-based implementations for missing statement types

Result<Unit, ExecutionError>
ReturnStmt::execute_result(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  INTERPRETER_DEBUG_LOG("DEBUG: ReturnStmt - Lane " << lane.laneId
            << " executing return (Result-based)");

  if (expr_) {
    #ifdef _MSC_VER
      auto _result = expr_->evaluate_result(lane, wave, tg);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
      Value exprResult = _result.unwrap();
    #else
      Value exprResult = TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
    #endif
    lane.returnValue = exprResult;
  }

  // Handle comprehensive global cleanup for early return
  handleGlobalEarlyReturn(lane, wave, tg);

  INTERPRETER_DEBUG_LOG("ReturnStmt - Return completed successfully");
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
BarrierStmt::execute_result(LaneContext &lane, WaveContext &wave,
                            ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Err<Unit, ExecutionError>(ExecutionError::InvalidState);

  INTERPRETER_DEBUG_LOG("Lane " << lane.laneId << " in wave " << wave.waveId
                                << " hitting barrier");

  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);

  // Create a new barrier or join existing one for collective execution
  uint32_t barrierId = tg.nextBarrierId;

  // Check if there's already an active barrier for this threadgroup
  bool foundActiveBarrier = false;
  for (auto &[id, barrier] : tg.activeBarriers) {
    // Join the first active barrier (since all threads must reach the same
    // barrier)
    barrierId = id;
    foundActiveBarrier = true;
    break;
  }

  if (!foundActiveBarrier) {
    // Create new barrier for collective execution
    ThreadgroupBarrierState newBarrier;
    newBarrier.barrierId = barrierId;

    // ALL threads in threadgroup must participate in barriers (even if some
    // returned early) This is different from wave operations which only include
    // active lanes Barriers require ALL threads that started execution to reach
    // the barrier
    for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
      for (size_t laneId = 0; laneId < tg.waves[waveId]->lanes.size();
           ++laneId) {
        // Include ALL threads, regardless of current state
        // If a thread returned early, it will trigger deadlock detection
        ThreadId participantTid = tg.getGlobalThreadId(waveId, laneId);
        newBarrier.participatingThreads.insert(participantTid);
      }
    }

    tg.activeBarriers[barrierId] = newBarrier;
    tg.nextBarrierId++;

    INTERPRETER_DEBUG_LOG("Created collective barrier "
                          << barrierId << " expecting "
                          << newBarrier.participatingThreads.size()
                          << " threads");
  }

  // Add this thread to the barrier for collective execution
  auto &barrier = tg.activeBarriers[barrierId];
  barrier.arrivedThreads.insert(tid);

  // Set thread state to waiting for collective barrier execution
  lane.state = ThreadState::WaitingAtBarrier;
  lane.waitingBarrierId = barrierId;

  INTERPRETER_DEBUG_LOG(
      "Thread " << tid << " waiting for collective barrier " << barrierId
                << " (" << barrier.arrivedThreads.size() << "/"
                << barrier.participatingThreads.size() << " arrived)");

  // If all threads have arrived, the barrier will execute collectively in
  // processBarriers()
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
SharedWriteStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  INTERPRETER_DEBUG_LOG("DEBUG: SharedWriteStmt - Lane " << lane.laneId
            << " executing shared write (Result-based)");

  #ifdef _MSC_VER
    auto _result = expr_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value value = _result.unwrap();
  #else
    auto value = TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  #endif
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  tg.sharedMemory->write(addr_, value, tid);

  INTERPRETER_DEBUG_LOG("DEBUG: SharedWriteStmt - Shared write completed successfully");
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string SharedReadExpr::toString() const {
  return "g_shared[" + std::to_string(addr_) + "]";
}

std::string BufferAccessExpr::toString() const {
  return bufferName_ + "[" + indexExpr_->toString() + "]";
}

std::string ArrayAccessExpr::toString() const {
  return arrayName_ + "[" + indexExpr_->toString() + "]";
}

std::string DispatchThreadIdExpr::toString() const {
  const char* components[] = {"x", "y", "z"};
  return "SV_DispatchThreadID." + std::string(components[component_]);
}

// MiniHLSLInterpreter implementation with cooperative scheduling
ExecutionResult MiniHLSLInterpreter::executeWithOrdering(
    const Program &program, const ThreadOrdering &ordering, uint32_t waveSize) {
  ExecutionResult result;

  // Use sequential ordering if no ordering is specified
  ThreadOrdering effectiveOrdering = ordering;
  if (ordering.executionOrder.empty()) {
    effectiveOrdering = ThreadOrdering::sequential(program.getTotalThreads());
  }

  // Create threadgroup context
  // Use the effective wave size from the program, which considers WaveSize attributes
  uint32_t effectiveWaveSize = program.getEffectiveWaveSize(waveSize);
  ThreadgroupContext tgContext(program.getTotalThreads(), effectiveWaveSize);
  tgContext.interpreter = this;  // Set interpreter pointer for callbacks
  
  // Initialize global buffers
  for (const auto& bufferDecl : program.globalBuffers) {
    auto buffer = std::make_shared<GlobalBuffer>(
        bufferDecl.size > 0 ? bufferDecl.size : 1024, // Default size if unbounded
        bufferDecl.bufferType);
    tgContext.globalBuffers[bufferDecl.name] = buffer;
  }
  
  // Initialize built-in variables for all lanes based on function parameters
  for (auto& wave : tgContext.waves) {
    for (auto& lane : wave->lanes) {
      initializeBuiltinVariables(*lane, *wave, tgContext, program);
    }
  }

  uint32_t orderingIndex = 0;
  uint32_t maxIterations = program.getTotalThreads() *
                           program.statements.size() * 1000; // Safety limit
  uint32_t iteration = 0;

  // Cooperative scheduling main loop
  while (iteration < maxIterations) {
    iteration++;

    // Process completed wave operations and barriers
    processWaveOperations(tgContext);
    // processBarriers(tgContext);

    // Process lanes waiting to resume control flow
    processControlFlowResumption(tgContext);

    // Get threads ready for execution
    auto readyThreads = tgContext.getReadyThreads();
    if (readyThreads.empty()) {
      // Check if all threads are completed
      bool allCompleted = true;
      bool hasWaitingThreads = false;
      for (const auto &wave : tgContext.waves) {
        for (const auto &lane : wave->lanes) {
          if (lane->state == ThreadState::WaitingForWave ||
              lane->state == ThreadState::WaitingAtBarrier ||
              lane->state == ThreadState::WaitingForResume) {
            hasWaitingThreads = true;
          }
          if (lane->state != ThreadState::Completed &&
              lane->state != ThreadState::Error) {
            allCompleted = false;
          }
        }
      }

      if (allCompleted) {
        // All threads have completed - safe to capture final state
        onExecutionComplete(tgContext);
        break; // All threads finished
      }

      // If we have waiting threads but no ready threads,
      // continue to let processWaveOperations wake them up
      if (hasWaitingThreads) {
        continue; // Let synchronization complete
      }

      // Check for true deadlock (no waiting, no ready, not completed)
      result.errorMessage = "Deadlock detected: no threads ready or waiting";
      break;
    }

    // Select next thread to execute according to ordering
    ThreadId nextTid = selectNextThread(readyThreads, effectiveOrdering, orderingIndex);

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

  // Collect final state
  result.sharedMemoryState = tgContext.sharedMemory->getSnapshot();

  // Collect global variables from first thread
  if (!tgContext.waves.empty() && !tgContext.waves[0]->lanes.empty()) {
    result.globalVariables = tgContext.waves[0]->lanes[0]->variables;
  }

  
  // Note: onExecutionComplete hook is now called inside the execution loop
  // when allCompleted is true, ensuring all threads have finished
  
  // Print Dynamic Block Execution Graph (DBEG) for debugging merge blocks
  if (ENABLE_BLOCK_DEBUG == true)
    tgContext.printDynamicExecutionGraph(true);

  // Print final variable values for all lanes
    if (ENABLE_BLOCK_DEBUG == true)
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
    INTERPRETER_DEBUG_LOG("lane " << lane.laneId << " have completed execution\n");
    lane.state = ThreadState::Completed;
    return true;
  }

  // Execute the current statement using Result-based approach with specialized
  // error handling
  const auto &stmt = program.statements[lane.currentStatement];

  // Use the specialized wrapper functions that handle errors properly
  auto result = stmt->execute_with_error_handling(lane, wave, tgContext);

  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();
    switch (error) {
    case ExecutionError::WaveOperationWait:
      // Lane is waiting for wave operation - scheduler will resume it later
      // Lane state should already be set to WaitingForWave by the wave
      // operation
      INTERPRETER_DEBUG_LOG("DEBUG: WAVE_WAIT: Lane " << (tid % 32)
                << " received WaveOperationWait error, state="
                << (int)lane.state);
      // Do nothing - state is already correct
      break;

    case ExecutionError::ControlFlowBreak:
    case ExecutionError::ControlFlowContinue:
      // These should have been handled by the wrapper functions
      // If we see them here, it means they propagated from nested statements
      // which indicates an error in our implementation
      lane.state = ThreadState::Error;
      lane.errorMessage = "Unhandled control flow error";
      break;

    case ExecutionError::InvalidState:
      // Runtime error occurred
      lane.state = ThreadState::Error;
      lane.errorMessage = "Invalid execution state";
      break;

    default:
      lane.state = ThreadState::Error;
      lane.errorMessage = "Unknown execution error";
      break;
    }
  } else {
    // Success case - handle statement completion
    if (lane.hasReturned) {
      lane.state = ThreadState::Completed;
    } else if (lane.state == ThreadState::Ready) {
      // Only increment if the lane is not waiting
      lane.currentStatement++;
    }
  }

  return true;
}

void MiniHLSLInterpreter::processWaveOperations(ThreadgroupContext &tgContext) {
  // Process wave operations for each wave
  for (size_t waveId = 0; waveId < tgContext.waves.size(); ++waveId) {
    auto &wave = *tgContext.waves[waveId];

    // Check active sync points for this wave
    std::vector<std::pair<const void *, uint32_t>> completedSyncPoints;

    // Find sync points that are ready to execute
    for (auto &[instructionKey, syncPoint] : wave.activeSyncPoints) {
      // Update sync point phase before checking
      syncPoint.updatePhase(tgContext, waveId);

      // Before checking if complete, update participants to include all current
      // lanes in the block
      if (syncPoint.shouldExecute(tgContext, waveId)) {
        // Update arrivedParticipants to include all lanes currently in this
        // block
        uint32_t blockId = instructionKey.second;
        const auto *block = tgContext.getBlock(blockId);
        if (block) {
          // Get participating lanes from registry (single source of truth)
          auto participatingLanes =
              tgContext.membershipRegistry.getParticipatingLanes(waveId,
                                                                 blockId);
          if (!participatingLanes.empty()) {
            // Update participants to include all current lanes in block
            for (LaneId laneId : participatingLanes) {
              syncPoint.arrivedParticipants.insert(laneId);
              syncPoint.expectedParticipants.insert(laneId);
            }
            INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Updated participants before "
                         "execution to include all current block lanes: ");
            for (LaneId laneId : syncPoint.arrivedParticipants) {
              INTERPRETER_DEBUG_LOG(laneId << " ");
            }
          }
        }
        INTERPRETER_DEBUG_LOG("Executing collective wave operation for wave "
                              << waveId << " at instruction "
                              << instructionKey.first << " block "
                              << instructionKey.second << "\n");

        // Execute the wave operation collectively
        executeCollectiveWaveOperation(tgContext, waveId, instructionKey,
                                       syncPoint);

        // Don't release yet - let lanes retrieve results first
      } else if (syncPoint.isReadyForCleanup()) {
        // Results are available, wake up lanes so they can retrieve them
        INTERPRETER_DEBUG_LOG(
            "Waking up lanes to retrieve wave operation results for wave "
            << waveId << "\n");
        for (LaneId laneId : syncPoint.arrivedParticipants) {
          if (laneId < wave.lanes.size() &&
              wave.lanes[laneId]->state == ThreadState::WaitingForWave) {
            INTERPRETER_DEBUG_LOG("  Waking up lane "
                                  << laneId
                                  << " from WaitingForWave to Ready\n");
            wave.lanes[laneId]->state = ThreadState::Ready;
            wave.lanes[laneId]->isResumingFromWaveOp =
                true; // Set resuming flag
            // Also remove from the waiting instruction map
            wave.laneWaitingAtInstruction.erase(laneId);

            // CRITICAL FIX: Update registry status from WaitingForWave back to
            // Participating Use lane's current block, not stale block from
            // instruction key
            uint32_t currentBlockId = tgContext.getCurrentBlock(waveId, laneId);
            tgContext.membershipRegistry.onLaneFinishWaveOp(waveId, laneId,
                                                            currentBlockId);
          }
        }

        // Mark for release after lanes retrieve results
        completedSyncPoints.push_back(instructionKey);
      } else if (syncPoint.expectedParticipants.empty()) {
        // No participants left - cleanup immediately
        INTERPRETER_DEBUG_LOG("DEBUG: processWaveOperations - Sync point "
                  << instructionKey.first << " in block "
                  << instructionKey.second
                  << " has no participants, cleaning up");
        completedSyncPoints.push_back(instructionKey);
      }
    }

    // Release completed sync points after lanes have retrieved their results
    for (const auto &instructionKey : completedSyncPoints) {
      // Only release if all results have been consumed
      auto &syncPoint = wave.activeSyncPoints[instructionKey];

      if (syncPoint.pendingResults.empty()) {
        INTERPRETER_DEBUG_LOG(
            "All results consumed, releasing sync point for wave "
            << waveId << " at instruction " << instructionKey.first << " block "
            << instructionKey.second << "\n");
        tgContext.releaseWaveSyncPoint(waveId, instructionKey);
      }
    }
  }
}

void MiniHLSLInterpreter::processControlFlowResumption(
    ThreadgroupContext &tgContext) {
  // Wake up lanes that are waiting to resume control flow statements
  for (size_t waveId = 0; waveId < tgContext.waves.size(); ++waveId) {
    auto &wave = *tgContext.waves[waveId];

    for (size_t laneId = 0; laneId < wave.lanes.size(); ++laneId) {
      auto &lane = *wave.lanes[laneId];

      if (lane.state == ThreadState::WaitingForResume) {
        // Make lane ready to resume control flow execution
        INTERPRETER_DEBUG_LOG("  Waking up lane "
                              << laneId << " from WaitingForResume to Ready\n");
        lane.state = ThreadState::Ready;
      }
    }
  }
}

Result<Unit, ExecutionError>
MiniHLSLInterpreter::executeCollectiveWaveOperation(
    ThreadgroupContext &tgContext, WaveId waveId,
    const std::pair<const void *, uint32_t> &instructionKey,
    WaveOperationSyncPoint &syncPoint) {
  // Get the actual WaveActiveOp object from the original instruction pointer
  // stored in syncPoint
  const WaveActiveOp *waveOp =
      static_cast<const WaveActiveOp *>(syncPoint.instruction);

  // Collect values from all participating lanes
  std::vector<Value> values;
  auto &wave = *tgContext.waves[waveId];

  for (LaneId laneId : syncPoint.arrivedParticipants) {
    if (laneId < wave.lanes.size()) {
      // Evaluate the expression for this lane
      #ifdef _MSC_VER
        auto _result = waveOp->getExpression()->evaluate_result(
                                   *wave.lanes[laneId], wave, tgContext);
        if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
        Value value = _result.unwrap();
      #else
      Value value = TRY_RESULT(waveOp->getExpression()->evaluate_result(
                                   *wave.lanes[laneId], wave, tgContext),
                               Unit, ExecutionError);
      #endif
      values.push_back(value);
    }
  }

  // Execute the wave operation on the collected values
  Value result = waveOp->computeWaveOperation(values);

  // Store the result for all participating lanes and transition to Executed
  // state
  INTERPRETER_DEBUG_LOG("DEBUG: WAVE_OP: Storing collective result for lanes: ");
  for (LaneId laneId : syncPoint.arrivedParticipants) {
    INTERPRETER_DEBUG_LOG(laneId << " ");
    syncPoint.pendingResults[laneId] = result;
  }
  
  // Call hook for wave operation sync point created
  onWaveOpSyncPointCreated(wave, tgContext, instructionKey.second, syncPoint.arrivedParticipants.size());
  
  // Call hook for wave operation executed
  onWaveOpExecuted(wave, tgContext, waveOp->toString(), result);
  
  // Mark sync point as executed using state machine
  syncPoint.markExecuted();
  INTERPRETER_DEBUG_LOG(" (phase: " << (int)syncPoint.getPhase() << ")");

  INTERPRETER_DEBUG_LOG(
      "Collective wave operation result: " << result.toString() << "\n");

  // Transition sync point state to Executed
  syncPoint.state = SyncPointState::Executed;

  return Ok<Unit, ExecutionError>(Unit{});
}

void MiniHLSLInterpreter::executeCollectiveBarrier(
    ThreadgroupContext &tgContext, uint32_t barrierId,
    const ThreadgroupBarrierState &barrier) {
  INTERPRETER_DEBUG_LOG("Executing collective barrier "
                        << barrierId << " with "
                        << barrier.arrivedThreads.size() << " threads");

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
  INTERPRETER_DEBUG_LOG("Collective barrier "
                        << barrierId << " memory fence and sync complete");
}

void MiniHLSLInterpreter::processBarriers(ThreadgroupContext &tgContext) {
  std::vector<uint32_t> completedBarriers;

  // Check each active barrier to see if it's complete or deadlocked
  for (auto &[barrierId, barrier] : tgContext.activeBarriers) {
    if (barrier.arrivedThreads.size() == barrier.participatingThreads.size()) {
      // All threads have arrived - execute barrier collectively
      INTERPRETER_DEBUG_LOG("Collective barrier "
                            << barrierId << " executing! All "
                            << barrier.arrivedThreads.size()
                            << " threads synchronized");

      // Execute barrier collectively (memory fence, synchronization point)
      executeCollectiveBarrier(tgContext, barrierId, barrier);

      // Release all waiting threads simultaneously
      for (ThreadId tid : barrier.arrivedThreads) {
        auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
        if (waveId < tgContext.waves.size() &&
            laneId < tgContext.waves[waveId]->lanes.size()) {
          auto &lane = tgContext.waves[waveId]->lanes[laneId];
          if (lane->state == ThreadState::WaitingAtBarrier &&
              lane->waitingBarrierId == barrierId) {
            lane->state = ThreadState::Ready;
            lane->waitingBarrierId = 0;
            // Advance to next statement after barrier completion
            lane->currentStatement++;
            INTERPRETER_DEBUG_LOG("Released thread "
                                  << tid << " from barrier " << barrierId
                                  << " and advanced to statement "
                                  << lane->currentStatement);
          }
        }
      }

      completedBarriers.push_back(barrierId);
    } else {
      // Check for deadlock - if some threads are completed/error but not all
      // expected threads arrived
      std::set<ThreadId> stillExecutingThreads;
      for (ThreadId tid : barrier.participatingThreads) {
        auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
        if (waveId < tgContext.waves.size() &&
            laneId < tgContext.waves[waveId]->lanes.size()) {
          auto &lane = tgContext.waves[waveId]->lanes[laneId];
          if (lane->state != ThreadState::Completed &&
              lane->state != ThreadState::Error) {
            stillExecutingThreads.insert(tid);
          }
        }
      }

      // If some threads that should participate are no longer executing, we
      // have a deadlock
      if (stillExecutingThreads.size() < barrier.participatingThreads.size() &&
          barrier.arrivedThreads.size() < stillExecutingThreads.size()) {
        // Deadlock detected - some threads completed without hitting barrier
        std::stringstream errorMsg;
        errorMsg
            << "Barrier deadlock detected! Barrier " << barrierId
            << " expected " << barrier.participatingThreads.size()
            << " threads, " << "but only " << barrier.arrivedThreads.size()
            << " arrived. "
            << "Some threads completed execution without reaching the barrier.";

        INTERPRETER_DEBUG_LOG("ERROR: " << errorMsg.str());

        // Mark all remaining threads as error state
        for (ThreadId tid : stillExecutingThreads) {
          auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
          if (waveId < tgContext.waves.size() &&
              laneId < tgContext.waves[waveId]->lanes.size()) {
            auto &lane = tgContext.waves[waveId]->lanes[laneId];
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
    verification.results.push_back(
        executeWithOrdering(program, ordering, waveSize));
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
  return std::make_unique<LiteralExpr>(v, HLSLType::Unknown);
}

std::unique_ptr<Expression> makeLiteral(Value v, const std::string& typeStr) {
  return std::make_unique<LiteralExpr>(v, typeStr);
}

std::unique_ptr<Expression> makeVariable(const std::string &name) {
  return std::make_unique<VariableExpr>(name, HLSLType::Unknown);
}

std::unique_ptr<Expression> makeVariable(const std::string &name, const std::string& typeStr) {
  return std::make_unique<VariableExpr>(name, typeStr);
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

std::unique_ptr<Statement> makeVarDeclWithType(const std::string &name,
                                               const std::string &type,
                                               std::unique_ptr<Expression> init) {
  return std::make_unique<VarDeclStmt>(name, type, std::move(init));
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

// Helper function implementations for HLSL type/semantic conversion
HLSLType HLSLTypeInfo::fromString(const std::string& typeStr) {
  static const std::unordered_map<std::string, HLSLType> typeMap = {
    {"bool", HLSLType::Bool}, {"int", HLSLType::Int}, {"uint", HLSLType::Uint},
    {"float", HLSLType::Float}, {"double", HLSLType::Double},
    {"bool2", HLSLType::Bool2}, {"bool3", HLSLType::Bool3}, {"bool4", HLSLType::Bool4},
    {"int2", HLSLType::Int2}, {"int3", HLSLType::Int3}, {"int4", HLSLType::Int4},
    {"uint2", HLSLType::Uint2}, {"uint3", HLSLType::Uint3}, {"uint4", HLSLType::Uint4},
    {"float2", HLSLType::Float2}, {"float3", HLSLType::Float3}, {"float4", HLSLType::Float4},
    {"double2", HLSLType::Double2}, {"double3", HLSLType::Double3}, {"double4", HLSLType::Double4},
    {"float2x2", HLSLType::Float2x2}, {"float3x3", HLSLType::Float3x3}, {"float4x4", HLSLType::Float4x4},
    {"float2x3", HLSLType::Float2x3}, {"float2x4", HLSLType::Float2x4},
    {"float3x2", HLSLType::Float3x2}, {"float3x4", HLSLType::Float3x4},
    {"float4x2", HLSLType::Float4x2}, {"float4x3", HLSLType::Float4x3},
    {"StructuredBuffer", HLSLType::StructuredBuffer},
    {"RWStructuredBuffer", HLSLType::RWStructuredBuffer}
  };
  
  auto it = typeMap.find(typeStr);
  return (it != typeMap.end()) ? it->second : HLSLType::Custom;
}

std::string HLSLTypeInfo::toString(HLSLType type) {
  switch (type) {
    case HLSLType::Bool: return "bool";
    case HLSLType::Int: return "int";
    case HLSLType::Uint: return "uint";
    case HLSLType::Float: return "float";
    case HLSLType::Double: return "double";
    case HLSLType::Bool2: return "bool2";
    case HLSLType::Bool3: return "bool3";
    case HLSLType::Bool4: return "bool4";
    case HLSLType::Int2: return "int2";
    case HLSLType::Int3: return "int3";
    case HLSLType::Int4: return "int4";
    case HLSLType::Uint2: return "uint2";
    case HLSLType::Uint3: return "uint3";
    case HLSLType::Uint4: return "uint4";
    case HLSLType::Float2: return "float2";
    case HLSLType::Float3: return "float3";
    case HLSLType::Float4: return "float4";
    case HLSLType::Double2: return "double2";
    case HLSLType::Double3: return "double3";
    case HLSLType::Double4: return "double4";
    case HLSLType::Float2x2: return "float2x2";
    case HLSLType::Float3x3: return "float3x3";
    case HLSLType::Float4x4: return "float4x4";
    case HLSLType::Float2x3: return "float2x3";
    case HLSLType::Float2x4: return "float2x4";
    case HLSLType::Float3x2: return "float3x2";
    case HLSLType::Float3x4: return "float3x4";
    case HLSLType::Float4x2: return "float4x2";
    case HLSLType::Float4x3: return "float4x3";
    case HLSLType::StructuredBuffer: return "StructuredBuffer";
    case HLSLType::RWStructuredBuffer: return "RWStructuredBuffer";
    default: return "unknown";
  }
}

bool HLSLTypeInfo::isVectorType(HLSLType type) {
  return type >= HLSLType::Bool2 && type <= HLSLType::Double4;
}

bool HLSLTypeInfo::isMatrixType(HLSLType type) {
  return type >= HLSLType::Float2x2 && type <= HLSLType::Float4x3;
}

uint32_t HLSLTypeInfo::getComponentCount(HLSLType type) {
  switch (type) {
    case HLSLType::Bool2: case HLSLType::Int2: case HLSLType::Uint2:
    case HLSLType::Float2: case HLSLType::Double2: return 2;
    case HLSLType::Bool3: case HLSLType::Int3: case HLSLType::Uint3:
    case HLSLType::Float3: case HLSLType::Double3: return 3;
    case HLSLType::Bool4: case HLSLType::Int4: case HLSLType::Uint4:
    case HLSLType::Float4: case HLSLType::Double4: return 4;
    default: return 1;
  }
}

HLSLSemantic HLSLSemanticInfo::fromString(const std::string& semanticStr) {
  static const std::unordered_map<std::string, HLSLSemantic> semanticMap = {
    {"SV_DispatchThreadID", HLSLSemantic::SV_DispatchThreadID},
    {"SV_GroupID", HLSLSemantic::SV_GroupID},
    {"SV_GroupThreadID", HLSLSemantic::SV_GroupThreadID},
    {"SV_GroupIndex", HLSLSemantic::SV_GroupIndex},
    {"SV_Position", HLSLSemantic::SV_Position},
    {"SV_VertexID", HLSLSemantic::SV_VertexID},
    {"SV_InstanceID", HLSLSemantic::SV_InstanceID},
    {"SV_Target", HLSLSemantic::SV_Target},
    {"SV_Depth", HLSLSemantic::SV_Depth},
    {"SV_Coverage", HLSLSemantic::SV_Coverage},
    {"SV_IsFrontFace", HLSLSemantic::SV_IsFrontFace},
    {"SV_PrimitiveID", HLSLSemantic::SV_PrimitiveID},
    {"SV_SampleIndex", HLSLSemantic::SV_SampleIndex}
  };
  
  if (semanticStr.empty()) return HLSLSemantic::None;
  auto it = semanticMap.find(semanticStr);
  return (it != semanticMap.end()) ? it->second : HLSLSemantic::Custom;
}

std::string HLSLSemanticInfo::toString(HLSLSemantic semantic) {
  switch (semantic) {
    case HLSLSemantic::SV_DispatchThreadID: return "SV_DispatchThreadID";
    case HLSLSemantic::SV_GroupID: return "SV_GroupID";
    case HLSLSemantic::SV_GroupThreadID: return "SV_GroupThreadID";
    case HLSLSemantic::SV_GroupIndex: return "SV_GroupIndex";
    case HLSLSemantic::SV_Position: return "SV_Position";
    case HLSLSemantic::SV_VertexID: return "SV_VertexID";
    case HLSLSemantic::SV_InstanceID: return "SV_InstanceID";
    case HLSLSemantic::SV_Target: return "SV_Target";
    case HLSLSemantic::SV_Depth: return "SV_Depth";
    case HLSLSemantic::SV_Coverage: return "SV_Coverage";
    case HLSLSemantic::SV_IsFrontFace: return "SV_IsFrontFace";
    case HLSLSemantic::SV_PrimitiveID: return "SV_PrimitiveID";
    case HLSLSemantic::SV_SampleIndex: return "SV_SampleIndex";
    case HLSLSemantic::None: return "";
    default: return "Custom";
  }
}

bool HLSLSemanticInfo::isComputeShaderSemantic(HLSLSemantic semantic) {
  return semantic == HLSLSemantic::SV_DispatchThreadID ||
         semantic == HLSLSemantic::SV_GroupID ||
         semantic == HLSLSemantic::SV_GroupThreadID ||
         semantic == HLSLSemantic::SV_GroupIndex;
}

// Helper function to extract parameter signature from AST
static ParameterSig extractParamSig(const clang::ParmVarDecl* P,
                                    clang::ASTContext& Ctx) {
  ParameterSig sig;
  sig.name = P->getNameAsString();
  
  // Get parameter type
  clang::QualType QT = P->getType();
  std::string typeStr = QT.getAsString(Ctx.getPrintingPolicy());
  sig.type = HLSLTypeInfo::fromString(typeStr);
  if (sig.type == HLSLType::Custom) {
    sig.customTypeStr = typeStr;
  }
  
  INTERPRETER_DEBUG_LOG("DEBUG: Parameter '" << sig.name << "' has type '" 
            << typeStr << "', mapped to enum " << (int)sig.type);
  
  // Initialize semantic to None
  sig.semantic = HLSLSemantic::None;
  sig.customSemanticStr = "";

  // Debug: print all attributes
  INTERPRETER_DEBUG_LOG("DEBUG: Parameter " << sig.name << " has " 
            << std::distance(P->attr_begin(), P->attr_end()) << " attributes");
  for (const auto* A : P->attrs()) {
    INTERPRETER_DEBUG_LOG("  Attr kind: " << A->getKind() 
              << ", spelling: " << A->getSpelling());
  }
  
  // 1) Best path: look for real HLSL semantic attribute
  if (const auto* SA = P->getAttr<clang::HLSLSemanticAttr>()) {
    // HLSLSemanticAttr has a 'name' argument based on Attr.td
    std::string semanticName = SA->getName();
    INTERPRETER_DEBUG_LOG("DEBUG: Found HLSLSemanticAttr with name: " << semanticName);
    
    sig.semantic = HLSLSemanticInfo::fromString(semanticName);
    if (sig.semantic == HLSLSemantic::Custom) {
      sig.customSemanticStr = semanticName;
    }
    return sig; // Found concrete semantic, we're done
  }
    
  // Some DXC versions have per-system-value attrs
  // Uncomment and adapt based on your DXC version:
  // if (llvm::isa<clang::HLSLSV_DispatchThreadIDAttr>(A)) {
  //   sig.semantic = HLSLSemantic::SV_DispatchThreadID;
  //   return sig;
  // }
  // if (llvm::isa<clang::HLSLSV_GroupIDAttr>(A)) {
  //   sig.semantic = HLSLSemantic::SV_GroupID;
  //   return sig;
  // }

  // 2) Since DXC doesn't expose semantics as attributes, use source text parsing
  // This is now the primary method for extracting semantics
  {
    clang::SourceManager& SM = Ctx.getSourceManager();
    clang::SourceRange R = P->getSourceRange();
    
    // Get the entire parameter declaration including semantic
    // We need to expand the range to capture text after the parameter name
    clang::SourceLocation StartLoc = R.getBegin();
    clang::SourceLocation EndLoc = R.getEnd();
    
    // Try to read more characters to capture the semantic
    const char* StartPtr = SM.getCharacterData(StartLoc);
    
    // Search for the semantic by looking for ':' after the parameter name
    std::string paramText;
    const char* ptr = StartPtr;
    int charCount = 0;
    // Read up to 200 characters to find the semantic
    while (ptr && *ptr && charCount < 200) {
      if (*ptr == ',' || *ptr == ')') {
        // Found end of this parameter
        break;
      }
      paramText += *ptr;
      ptr++;
      charCount++;
    }
    
    INTERPRETER_DEBUG_LOG("DEBUG: Raw parameter text for '" << sig.name << "': " << paramText);
    
    // Look for semantic after ':'
    if (auto pos = paramText.find(':'); pos != std::string::npos) {
      std::string after = paramText.substr(pos + 1);
      // Trim whitespace
      size_t start = after.find_first_not_of(" \t\n\r");
      if (start != std::string::npos) {
        size_t end = after.find_first_of(" \t\n\r),;");
        std::string semanticStr = (end == std::string::npos) 
          ? after.substr(start) 
          : after.substr(start, end - start);
          
        INTERPRETER_DEBUG_LOG("DEBUG: Extracted semantic string: '" << semanticStr << "'");
        
        sig.semantic = HLSLSemanticInfo::fromString(semanticStr);
        if (sig.semantic == HLSLSemantic::Custom) {
          sig.customSemanticStr = semanticStr;
        }
      }
    }
  }

  return sig;
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


  // First, parse global buffer declarations from the translation unit
  if (auto* TU = context.getTranslationUnitDecl()) {
    for (auto* Decl : TU->decls()) {
      if (auto* VD = clang::dyn_cast<clang::VarDecl>(Decl)) {
        // Check if this is a buffer declaration
        std::string typeName = VD->getType().getAsString();
        if (typeName.find("RWBuffer") != std::string::npos ||
            typeName.find("Buffer") != std::string::npos ||
            typeName.find("StructuredBuffer") != std::string::npos ||
            typeName.find("ByteAddressBuffer") != std::string::npos ||
            typeName.find("Texture") != std::string::npos) {
          
          interpreter::GlobalBufferDecl bufferDecl;
          bufferDecl.name = VD->getName().str();
          bufferDecl.isReadWrite = (typeName.find("RW") != std::string::npos);
          
          // Extract buffer type
          if (typeName.find("RWStructuredBuffer") != std::string::npos) {
            bufferDecl.bufferType = "RWStructuredBuffer";
          } else if (typeName.find("RWByteAddressBuffer") != std::string::npos) {
            bufferDecl.bufferType = "RWByteAddressBuffer";
          } else if (typeName.find("RWBuffer") != std::string::npos) {
            bufferDecl.bufferType = "RWBuffer";
          } else if (typeName.find("RWTexture") != std::string::npos) {
            bufferDecl.bufferType = "RWTexture";
          } else if (typeName.find("StructuredBuffer") != std::string::npos) {
            bufferDecl.bufferType = "StructuredBuffer";
          } else if (typeName.find("ByteAddressBuffer") != std::string::npos) {
            bufferDecl.bufferType = "ByteAddressBuffer";
          } else if (typeName.find("Texture") != std::string::npos) {
            bufferDecl.bufferType = "Texture";
          } else {
            bufferDecl.bufferType = "Buffer";
          }
          
          // Extract element type (simplified - assumes uint for now)
          bufferDecl.elementType = HLSLType::Uint;
          bufferDecl.size = 0; // Unbounded
          bufferDecl.registerIndex = 0; // Default to u0/t0
          
          // TODO: Check for register annotation once HLSLResourceBindingAttr is available
          
          
          result.program.globalBuffers.push_back(bufferDecl);
        }
      }
    }
  }

  try {
    // Extract thread configuration from function attributes
    extractThreadConfiguration(func, result.program);

    // Parse function parameters and semantics
    result.program.entryInputs.parameters.reserve(func->getNumParams());
    for (const clang::ParmVarDecl* P : func->parameters()) {
      auto sig = extractParamSig(P, context);
      
      // Handle special system value semantics
      if (sig.semantic == HLSLSemantic::SV_DispatchThreadID) {
        result.program.entryInputs.dispatchThreadIdParamName = sig.name;
        result.program.entryInputs.hasDispatchThreadID = true;
        
        // Validate it's a 3-component vector
        if (sig.type != HLSLType::Uint3 && sig.type != HLSLType::Int3) {
          result.errorMessage = "SV_DispatchThreadID must be uint3 or int3, got type " + std::to_string((int)sig.type);
          return result;
        }
      } else if (sig.semantic == HLSLSemantic::SV_GroupID) {
        result.program.entryInputs.groupIdParamName = sig.name;
        result.program.entryInputs.hasGroupID = true;
        // Similar validation could be added for other semantics
      } else if (sig.semantic == HLSLSemantic::SV_GroupThreadID) {
        result.program.entryInputs.groupThreadIdParamName = sig.name;
        result.program.entryInputs.hasGroupThreadID = true;
      }
      
      // Check if parameter is a struct - if so, check field semantics
      if (const auto* RT = P->getType()->getAs<clang::RecordType>()) {
        if (const auto* RD = RT->getDecl()) {
          INTERPRETER_DEBUG_LOG("Parameter " << sig.name << " is a struct, checking field semantics...");
          for (const auto* Field : RD->fields()) {
            if (const auto* FSA = Field->getAttr<clang::HLSLSemanticAttr>()) {
              std::string fieldSemantic = FSA->getName();
              INTERPRETER_DEBUG_LOG("  Field " << Field->getNameAsString() 
                        << " has semantic: " << fieldSemantic);
              // TODO: Store field semantics in a more structured way
            }
          }
        }
      }
      
      result.program.entryInputs.parameters.push_back(sig);
      
      // Log parameter information
      INTERPRETER_DEBUG_LOG("Parameter: " << HLSLTypeInfo::toString(sig.type) 
                << " " << sig.name);
      if (sig.semantic != HLSLSemantic::None) {
        INTERPRETER_DEBUG_LOG(" : " << HLSLSemanticInfo::toString(sig.semantic));
      }
    }
    
    // Check for return type semantic (e.g., SV_Target for pixel shaders)
    if (const auto* RSA = func->getAttr<clang::HLSLSemanticAttr>()) {
      std::string returnSemantic = RSA->getName();
      INTERPRETER_DEBUG_LOG("Function has return semantic: " << returnSemantic);
      // TODO: Store return semantic in program structure
    }

    // Get the function body
    const clang::CompoundStmt *body =
        clang::dyn_cast<clang::CompoundStmt>(func->getBody());
    if (!body) {
      result.errorMessage = "Function body is not a compound statement";
      return result;
    }

    // Convert the function body to interpreter statements
    convertCompoundStatement(body, result.program, context);

    INTERPRETER_DEBUG_LOG("Converted AST to interpreter program with "
              << result.program.statements.size() << " statements");

    result.success = true;
    return result;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Exception during conversion: ") + e.what();
    return result;
  }
}

// AST traversal helper methods
// Initialize built-in variables based on function parameters
void initializeBuiltinVariables(LaneContext& lane, 
                                                    WaveContext& wave,
                                                    ThreadgroupContext& tg,
                                                    const Program& program) {
  // Calculate thread IDs based on wave and lane
  uint32_t globalTid = wave.waveId * wave.waveSize + lane.laneId;
  
  // For each parameter with a system semantic, create the appropriate variable
  for (const auto& param : program.entryInputs.parameters) {
    switch (param.semantic) {
      case HLSLSemantic::SV_DispatchThreadID:
        // For 3D dispatch, we'd need dispatch dimensions. For now, assume 1D.
        if (param.type == HLSLType::Uint3) {
          // Create a struct-like representation or individual components
          lane.variables[param.name + ".x"] = Value(static_cast<int>(globalTid));
          lane.variables[param.name + ".y"] = Value(0);
          lane.variables[param.name + ".z"] = Value(0);
          // Also store as the full variable name for direct access
          lane.variables[param.name] = Value(static_cast<int>(globalTid)); // x component
        }
        break;
        
      case HLSLSemantic::SV_GroupID:
        if (param.type == HLSLType::Uint3) {
          lane.variables[param.name + ".x"] = Value(static_cast<int>(wave.waveId));
          lane.variables[param.name + ".y"] = Value(0);
          lane.variables[param.name + ".z"] = Value(0);
          lane.variables[param.name] = Value(static_cast<int>(wave.waveId));
        }
        break;
        
      case HLSLSemantic::SV_GroupThreadID:
        if (param.type == HLSLType::Uint3) {
          lane.variables[param.name + ".x"] = Value(static_cast<int>(lane.laneId));
          lane.variables[param.name + ".y"] = Value(0);
          lane.variables[param.name + ".z"] = Value(0);
          lane.variables[param.name] = Value(static_cast<int>(lane.laneId));
        }
        break;
        
      case HLSLSemantic::SV_GroupIndex:
        if (param.type == HLSLType::Uint) {
          lane.variables[param.name] = Value(static_cast<int>(lane.laneId));
        }
        break;
        
      default:
        // Non-system semantics or custom semantics - no automatic initialization
        break;
    }
  }
  
  INTERPRETER_DEBUG_LOG("DEBUG: Initialized built-in variables for lane " << lane.laneId);
  for (const auto& var : lane.variables) {
    INTERPRETER_DEBUG_LOG("  " << var.first << " = " << var.second.toString());
  }
}

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
    INTERPRETER_DEBUG_LOG("Found numthreads attribute: [" << program.numThreadsX << ", "
              << program.numThreadsY << ", " << program.numThreadsZ << "]");
  } else {
    INTERPRETER_DEBUG_LOG("No numthreads attribute found, using default [1, 1, 1]");
  }
  
  // Look for HLSLWaveSizeAttr
  if (const clang::HLSLWaveSizeAttr *attr =
          func->getAttr<clang::HLSLWaveSizeAttr>()) {
    // For simplicity, use the preferred size if specified, otherwise use min
    if (attr->getPreferred() > 0) {
      program.waveSize = attr->getPreferred();
    } else if (attr->getMin() > 0) {
      program.waveSize = attr->getMin();
    }
    INTERPRETER_DEBUG_LOG("Found WaveSize attribute: " << program.waveSize);
  }
}

void MiniHLSLInterpreter::convertCompoundStatement(
    const clang::CompoundStmt *compound, Program &program,
    clang::ASTContext &context) {

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
  } else if (auto expr = clang::dyn_cast<clang::Expr>(stmt)) {
    // Handle other expressions as statements
    if (auto compoundOp =
            clang::dyn_cast<clang::CompoundAssignOperator>(expr)) {
      return convertCompoundAssignOperator(compoundOp, context);
    }
    // For other expressions, wrap in ExprStmt
    auto convertedExpr = convertExpression(expr, context);
    if (convertedExpr) {
      return std::make_unique<ExprStmt>(std::move(convertedExpr));
    }
    return nullptr;
  } else if (auto compound = clang::dyn_cast<clang::CompoundStmt>(stmt)) {
    // Nested compound statement - this should not happen in our current design
    INTERPRETER_DEBUG_LOG("Warning: nested compound statement found, skipping");
    return nullptr;
  } else {
    INTERPRETER_DEBUG_LOG("Unsupported statement type: " << stmt->getStmtClassName());
    return nullptr;
  }
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertBinaryOperator(const clang::BinaryOperator *binOp,
                                           clang::ASTContext &context) {
  if (!binOp->isAssignmentOp()) {
    INTERPRETER_DEBUG_LOG("Non-assignment binary operator, skipping");
    return nullptr;
  }

  // Handle assignment: LHS = RHS
  auto lhs = convertExpression(binOp->getLHS(), context);
  auto rhs = convertExpression(binOp->getRHS(), context);

  if (!rhs) {
    PARSER_DEBUG_LOG("Failed to convert assignment RHS");
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertCompoundAssignOperator(
    const clang::CompoundAssignOperator *compoundOp,
    clang::ASTContext &context) {

  // Get the target variable name
  std::string targetVar = "unknown";
  if (auto declRef =
          clang::dyn_cast<clang::DeclRefExpr>(compoundOp->getLHS())) {
    targetVar = declRef->getDecl()->getName().str();
  }

  // Convert compound assignment to regular assignment
  // For example: result += 1 becomes result = result + 1
  auto lhs = convertExpression(compoundOp->getLHS(), context);
  auto rhs = convertExpression(compoundOp->getRHS(), context);

  if (!lhs || !rhs) {
    PARSER_DEBUG_LOG("Failed to convert compound assignment operands");
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
    INTERPRETER_DEBUG_LOG("Unsupported compound assignment operator");
    return nullptr;
  }

  // Create the binary expression: lhs op rhs
  auto binaryExpr =
      std::make_unique<BinaryOpExpr>(std::move(lhs), std::move(rhs), opType);

  INTERPRETER_DEBUG_LOG("DEBUG: Creating AssignStmt for compound assignment: "
            << targetVar << " = " << binaryExpr->toString());

  auto assignStmt = makeAssign(targetVar, std::move(binaryExpr));
  INTERPRETER_DEBUG_LOG("DEBUG: Created statement type: AssignStmt with toString: "
            << assignStmt->toString());

  return assignStmt;
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertCallExpression(const clang::CallExpr *callExpr,
                                           clang::ASTContext &context) {
  if (auto funcDecl = callExpr->getDirectCallee()) {
    std::string funcName = funcDecl->getName().str();

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
    } else if (funcName == "WaveActiveAllEqual" &&
               callExpr->getNumArgs() == 1) {
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
    INTERPRETER_DEBUG_LOG("Unsupported function call: " << funcName);
  }

  return nullptr;
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertDeclarationStatement(
    const clang::DeclStmt *declStmt, clang::ASTContext &context) {

  for (const auto *decl : declStmt->decls()) {
    if (auto varDecl = clang::dyn_cast<clang::VarDecl>(decl)) {
      std::string varName = varDecl->getName().str();
      
      // Get the actual type from Clang AST
      clang::QualType qualType = varDecl->getType();
      std::string typeName = qualType.getAsString();
      
      
      // Create a variable declaration with type information
      if (varDecl->hasInit()) {
        auto initExpr = convertExpression(varDecl->getInit(), context);
        if (initExpr) {
          return makeVarDeclWithType(varName, typeName, std::move(initExpr));
        }
      } else {
        // Variable declaration without initializer
        return makeVarDeclWithType(varName, typeName, nullptr);
      }
    }
  }

  return nullptr;
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertIfStatement(const clang::IfStmt *ifStmt,
                                        clang::ASTContext &context) {

  // Convert the condition expression
  auto condition = convertExpression(ifStmt->getCond(), context);
  if (!condition) {
    PARSER_DEBUG_LOG("Failed to convert if condition");
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
      INTERPRETER_DEBUG_LOG("For loop init is not a declaration statement");
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
    INTERPRETER_DEBUG_LOG("Converting for loop increment expression\n");
    increment = convertExpression(incExpr, context);
    if (!increment) {
      PARSER_DEBUG_LOG("Failed to convert for loop increment expression\n");
    } else {
      PARSER_DEBUG_LOG("Successfully converted increment to: " << increment->toString() << "\n");
    }
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
    INTERPRETER_DEBUG_LOG("For loop missing required components (var: " << loopVar
              << ", init: " << (init ? "yes" : "no")
              << ", condition: " << (condition ? "yes" : "no")
              << ", increment: " << (increment ? "yes" : "no") << ")");
    return nullptr;
  }

  return std::make_unique<ForStmt>(loopVar, std::move(init),
                                   std::move(condition), std::move(increment),
                                   std::move(body));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertWhileStatement(const clang::WhileStmt *whileStmt,
                                           clang::ASTContext &context) {

  // Convert the condition expression
  auto condition = convertExpression(whileStmt->getCond(), context);
  if (!condition) {
    PARSER_DEBUG_LOG("Failed to convert while condition");
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
    PARSER_DEBUG_LOG("Failed to convert do-while condition");
    return nullptr;
  }

  return std::make_unique<DoWhileStmt>(std::move(body), std::move(condition));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertSwitchStatement(const clang::SwitchStmt *switchStmt,
                                            clang::ASTContext &context) {

  // Convert the condition expression
  auto condition = convertExpression(switchStmt->getCond(), context);
  if (!condition) {
    PARSER_DEBUG_LOG("Failed to convert switch condition");
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
        INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - processing AST node: "
                  << stmt->getStmtClassName());
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
            INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - LHS type: "
                      << lhs->getStmtClassName());
            // Handle implicit casts
            auto unwrapped = lhs->IgnoreImpCasts();
            if (auto intLit =
                    clang::dyn_cast<clang::IntegerLiteral>(unwrapped)) {
              currentCaseValue = intLit->getValue().getSExtValue();
              INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - found case "
                        << currentCaseValue.value());
            } else {
              INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - unwrapped LHS type: "
                        << unwrapped->getStmtClassName());
            }
          } else {
            INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - no LHS found");
          }

          // Handle nested case statements (e.g., case 2: case 3: stmt)
          auto substmt = caseStmt->getSubStmt();
          while (substmt) {
            if (auto nestedCase = clang::dyn_cast<clang::CaseStmt>(substmt)) {
              // This is a nested case - save current case as empty
              if (currentCaseValue.has_value()) {
                INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - saving empty case "
                          << currentCaseValue.value()
                          << " (falls through to next)");
                switchResult->addCase(
                    currentCaseValue.value(),
                    std::vector<std::unique_ptr<Statement>>());
              }
              // Process the nested case
              if (auto lhs = nestedCase->getLHS()) {
                auto unwrapped = lhs->IgnoreImpCasts();
                if (auto intLit =
                        clang::dyn_cast<clang::IntegerLiteral>(unwrapped)) {
                  currentCaseValue = intLit->getValue().getSExtValue();
                  INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - found nested case "
                            << currentCaseValue.value());
                }
              }
              substmt = nestedCase->getSubStmt();
            } else {
              // This is the actual statement for the case
              if (auto converted = convertStatement(substmt, context)) {
                currentCase.push_back(std::move(converted));
              }
              break;
            }
          }
        } else if (auto defaultStmt =
                       clang::dyn_cast<clang::DefaultStmt>(stmt)) {
          INTERPRETER_DEBUG_LOG("DEBUG: Switch parsing - found default case");
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
  return std::make_unique<BreakStmt>();
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertContinueStatement(
    const clang::ContinueStmt *continueStmt, clang::ASTContext &context) {
  return std::make_unique<ContinueStmt>();
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertReturnStatement(const clang::ReturnStmt *returnStmt,
                                            clang::ASTContext &context) {

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

  // Extract type information from the expression
  std::string exprType = expr->getType().getAsString();
  

  // Handle different expression types
  // Check CompoundAssignOperator first since it inherits from BinaryOperator
  if (auto compoundOp = clang::dyn_cast<clang::CompoundAssignOperator>(expr)) {
    // For compound assignment in expression context (e.g., i0 += 1 in for loop)
    // Convert it to the equivalent binary expression: i0 + 1
    auto lhs = convertExpression(compoundOp->getLHS(), context);
    auto rhs = convertExpression(compoundOp->getRHS(), context);
    
    if (!lhs || !rhs) {
      PARSER_DEBUG_LOG("Failed to convert compound assignment operands\n");
      return nullptr;
    }
    
    // Map compound assignment to binary operation
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
      INTERPRETER_DEBUG_LOG("Unsupported compound assignment in expression context\n");
      return nullptr;
    }
    
    // Return the binary expression (without the assignment side effect)
    // The ForStmt will handle the actual assignment
    return makeBinaryOp(std::move(lhs), std::move(rhs), opType);
  } else if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(expr)) {
    auto result = convertBinaryExpression(binOp, context);
    if (result) {
      HLSLType hlslType = HLSLTypeInfo::fromString(exprType);
      if (hlslType == HLSLType::Custom) {
        result->setCustomType(exprType);
      } else {
        result->setType(hlslType);
      }
    }
    return result;
  } else if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) {
    std::string varName = declRef->getDecl()->getName().str();
    return makeVariable(varName, exprType);
  } else if (auto intLit = clang::dyn_cast<clang::IntegerLiteral>(expr)) {
    int64_t value = intLit->getValue().getSExtValue();
    return makeLiteral(Value(static_cast<int>(value)), exprType);
  } else if (auto floatLit = clang::dyn_cast<clang::FloatingLiteral>(expr)) {
    double value = floatLit->getValueAsApproximateDouble();
    return makeLiteral(Value(static_cast<float>(value)), exprType);
  } else if (auto boolLit = clang::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
    bool value = boolLit->getValue();
    return makeLiteral(Value(value), exprType);
  } else if (auto parenExpr = clang::dyn_cast<clang::ParenExpr>(expr)) {
    return convertExpression(parenExpr->getSubExpr(), context);
  } else if (auto implicitCast =
                 clang::dyn_cast<clang::ImplicitCastExpr>(expr)) {
    return convertExpression(implicitCast->getSubExpr(), context);
  } else if (auto operatorCall =
                 clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    auto result = convertOperatorCall(operatorCall, context);
    if (result) {
      HLSLType hlslType = HLSLTypeInfo::fromString(exprType);
      if (hlslType == HLSLType::Custom) {
        result->setCustomType(exprType);
      } else {
        result->setType(hlslType);
      }
    }
    return result;
  } else if (auto callExpr = clang::dyn_cast<clang::CallExpr>(expr)) {
    auto result = convertCallExpressionToExpression(callExpr, context);
    if (result) {
      HLSLType hlslType = HLSLTypeInfo::fromString(exprType);
      if (hlslType == HLSLType::Custom) {
        result->setCustomType(exprType);
      } else {
        result->setType(hlslType);
      }
    }
    return result;
  } else if (auto condOp = clang::dyn_cast<clang::ConditionalOperator>(expr)) {
    auto result = convertConditionalOperator(condOp, context);
    if (result) {
      HLSLType hlslType = HLSLTypeInfo::fromString(exprType);
      if (hlslType == HLSLType::Custom) {
        result->setCustomType(exprType);
      } else {
        result->setType(hlslType);
      }
    }
    return result;
  } else if (auto unaryOp = clang::dyn_cast<clang::UnaryOperator>(expr)) {
    auto result = convertUnaryExpression(unaryOp, context);
    if (result) {
      HLSLType hlslType = HLSLTypeInfo::fromString(exprType);
      if (hlslType == HLSLType::Custom) {
        result->setCustomType(exprType);
      } else {
        result->setType(hlslType);
      }
    }
    return result;
  } else if (auto vecElem = clang::dyn_cast<clang::HLSLVectorElementExpr>(expr)) {
    // Handle vector element access like tid.x, tid.y, tid.z
    auto baseExpr = vecElem->getBase();
    if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(baseExpr)) {
      std::string varName = declRef->getDecl()->getName().str();
      std::string accessor = vecElem->getAccessor().getName().str();
      
      // Create a variable expression for the vector element access
      std::string fullName = varName + "." + accessor;
      return std::make_unique<VariableExpr>(fullName, HLSLTypeInfo::fromString(exprType));
    }
    // For other vector accesses, try to evaluate the base and handle as needed
    INTERPRETER_DEBUG_LOG("Unsupported vector element access pattern");
    return nullptr;
  } else {
    INTERPRETER_DEBUG_LOG("Unsupported expression type: " << expr->getStmtClassName());
    return nullptr;
  }
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertUnaryExpression(const clang::UnaryOperator *unaryOp,
                                            clang::ASTContext &context) {
  auto operand = convertExpression(unaryOp->getSubExpr(), context);
  if (!operand) {
    PARSER_DEBUG_LOG("Failed to convert unary operand");
    return nullptr;
  }

  // Map Clang unary operator to interpreter unary operator
  switch (unaryOp->getOpcode()) {
  case clang::UO_PreInc: // ++i
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::PreIncrement);
  case clang::UO_PostInc: // i++
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::PostIncrement);
  case clang::UO_PreDec: // --i
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::PreDecrement);
  case clang::UO_PostDec: // i--
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::PostDecrement);
  case clang::UO_Plus: // +expr
    return std::make_unique<UnaryOpExpr>(std::move(operand), UnaryOpExpr::Plus);
  case clang::UO_Minus: // -expr
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::Minus);
  case clang::UO_Not: // !expr
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::LogicalNot);
  case clang::UO_LNot: // !expr (logical not)
    return std::make_unique<UnaryOpExpr>(std::move(operand),
                                         UnaryOpExpr::LogicalNot);
  default:
    INTERPRETER_DEBUG_LOG("Unsupported unary operator: "
              << unaryOp->getOpcodeStr(unaryOp->getOpcode()).str());
    return nullptr;
  }
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertCallExpressionToExpression(
    const clang::CallExpr *callExpr, clang::ASTContext &context) {
  if (auto funcDecl = callExpr->getDirectCallee()) {
    std::string funcName = funcDecl->getName().str();

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
    } else if (funcName == "WaveReadLaneAt" && callExpr->getNumArgs() == 2) {
      auto value = convertExpression(callExpr->getArg(0), context);
      auto laneIndex = convertExpression(callExpr->getArg(1), context);
      if (value && laneIndex) {
        // Get the type from the value expression
        HLSLType type = value->getType();
        return std::make_unique<WaveReadLaneAt>(std::move(value), 
                                                std::move(laneIndex), 
                                                type);
      }
    }

    INTERPRETER_DEBUG_LOG("Unsupported function call in expression context: " << funcName);
  }

  return nullptr;
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertConditionalOperator(
    const clang::ConditionalOperator *condOp, clang::ASTContext &context) {

  // Convert the condition
  auto condition = convertExpression(condOp->getCond(), context);
  if (!condition) {
    PARSER_DEBUG_LOG("Failed to convert conditional operator condition");
    return nullptr;
  }

  // Convert the true expression
  auto trueExpr = convertExpression(condOp->getTrueExpr(), context);
  if (!trueExpr) {
    PARSER_DEBUG_LOG("Failed to convert conditional operator true expression");
    return nullptr;
  }

  // Convert the false expression
  auto falseExpr = convertExpression(condOp->getFalseExpr(), context);
  if (!falseExpr) {
    PARSER_DEBUG_LOG("Failed to convert conditional operator false expression");
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
  case clang::BO_And:
    opType = BinaryOpExpr::BitwiseAnd;
    break;
  case clang::BO_Or:
    opType = BinaryOpExpr::BitwiseOr;
    break;
  case clang::BO_Xor:
    opType = BinaryOpExpr::Xor;
    break;
  case clang::BO_Assign:
    // Assignment in expression context (e.g., in for loop increment)
    if (auto varExpr = clang::dyn_cast<clang::DeclRefExpr>(binOp->getLHS())) {
      std::string varName = varExpr->getDecl()->getName().str();
      return std::make_unique<AssignExpr>(varName, std::move(rhs));
    }
    INTERPRETER_DEBUG_LOG("Assignment expression with non-variable LHS\n");
    return nullptr;
  default:
    INTERPRETER_DEBUG_LOG("Unsupported binary operator\n");
    return nullptr;
  }

  return makeBinaryOp(std::move(lhs), std::move(rhs), opType);
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertOperatorCall(
    const clang::CXXOperatorCallExpr *opCall, clang::ASTContext &context) {
  clang::OverloadedOperatorKind op = opCall->getOperator();

  if (op == clang::OO_Subscript) {
    // Array access: buffer[index]

    if (opCall->getNumArgs() >= 2) {
      // Get the base expression (the array/buffer being accessed)
      auto baseExpr = opCall->getArg(0);
      std::string bufferName;

      // Try to extract buffer name from DeclRefExpr
      if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(baseExpr)) {
        bufferName = declRef->getDecl()->getName().str();
        INTERPRETER_DEBUG_LOG("Array access on buffer: " << bufferName);
      } else {
        INTERPRETER_DEBUG_LOG("Complex base expression for array access");
        bufferName = "unknown_buffer";
      }

      // Convert the index expression
      auto indexExpr = convertExpression(opCall->getArg(1), context);
      if (indexExpr) {
        // Create a buffer access expression for global buffers (RWBuffer, etc.)
        return std::make_unique<BufferAccessExpr>(bufferName,
                                                  std::move(indexExpr));
      }
    }
  }

  INTERPRETER_DEBUG_LOG("Unsupported operator call");
  return nullptr;
}

// ConditionalExpr implementation
ConditionalExpr::ConditionalExpr(std::unique_ptr<Expression> condition,
                                 std::unique_ptr<Expression> trueExpr,
                                 std::unique_ptr<Expression> falseExpr)
    : Expression(""), condition_(std::move(condition)), trueExpr_(std::move(trueExpr)),
      falseExpr_(std::move(falseExpr)) {}

bool ConditionalExpr::isDeterministic() const {
  return condition_->isDeterministic() && trueExpr_->isDeterministic() &&
         falseExpr_->isDeterministic();
}

std::string ConditionalExpr::toString() const {
  return "(" + condition_->toString() + " ? " + trueExpr_->toString() + " : " +
         falseExpr_->toString() + ")";
}

// WhileStmt implementation
WhileStmt::WhileStmt(std::unique_ptr<Expression> cond,
                     std::vector<std::unique_ptr<Statement>> body)
    : condition_(std::move(cond)), body_(std::move(body)) {}

// Pure Result-based WhileStmt phase implementations
Result<bool, ExecutionError>
WhileStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)");

  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<bool, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), bool,
                             ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError> WhileStmt::executeBody(LaneContext &lane,
                                                    WaveContext &wave,
                                                    ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " executing body (Result-based)");

  // Execute body statements using Result-based approach
  for (const auto &stmt : body_) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Break encountered in body");
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Continue encountered in body");
        return Ok<Unit, ExecutionError>(Unit{}); // Continue to next iteration
      } else {
        // Other errors (like WaveOperationWait) should be propagated
        return result;
      }
    }
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> WhileStmt::execute_result(LaneContext &lane,
                                                       WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting")
            << " Result-based while loop");

  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;

  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId,
                        mergeBlockId);
    ourStackIndex = lane.executionStack.size() - 1;
  } else {
    // Get our execution state
    auto &ourEntry = lane.executionStack[ourStackIndex];
    headerBlockId = ourEntry.loopHeaderBlockId;
    mergeBlockId = ourEntry.loopMergeBlockId;
  }

  // Get our execution state
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Result-based state machine for while loop execution
  switch (ourEntry.phase) {
  case LaneContext::ControlFlowPhase::EvaluatingCondition: {
    // Evaluate condition using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId),
               Unit, ExecutionError);
    #endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::ExecutingBody: {
    INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
              << " executing body for iteration " << ourEntry.loopIteration
              << " from statement " << ourEntry.statementIndex
              << " (Result-based)");

    // Debug: Print current variable state
    INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
              << " variables at body start: ");
    for (const auto &var : lane.variables) {
      INTERPRETER_DEBUG_LOG(var.first << "=" << var.second.toString() << " ");
    }

    // Only set up iteration blocks if we're starting the body (statement index 0)
    if (ourEntry.statementIndex == 0) {
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);
    }

    // Execute body statements using Result-based helper method
    #ifdef _MSC_VER
      auto _result = executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId),
               Unit, ExecutionError);
    #endif

    // Check if we need to return early (lane returned or needs resume)
    if (lane.hasReturned || lane.state != ThreadState::Ready) {
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Clean up after body execution using non-throwing helper method
    cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::Reconverging: {
    // Handle loop exit using non-throwing helper method
    handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  default:
    INTERPRETER_DEBUG_LOG("ERROR: WhileStmt - Lane " << lane.laneId
              << " unexpected phase in Result-based execution");
    lane.executionStack.pop_back();
    return Ok<Unit, ExecutionError>(Unit{});
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

// Specialized wrapper function for WhileStmt-specific error handling
Result<Unit, ExecutionError>
WhileStmt::execute_with_error_handling(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg) {
  auto result = execute_result(lane, wave, tg);
  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();

    switch (error) {
    case ExecutionError::WaveOperationWait:
      // WhileStmt-specific: Wave operation already set the lane to
      // WaitingForWave Just propagate the error without changing state
      return result; // Propagate for parent to handle

    case ExecutionError::ControlFlowBreak:
      // WhileStmt consumes break: exit loop cleanly
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId);
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - break handled

    case ExecutionError::ControlFlowContinue:
      // WhileStmt consumes continue: jump back to condition phase
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - continue handled

    default:
      return result; // Propagate unknown errors
    }
  }
  return result; // Success case
}

// Helper method for fresh execution setup in WhileStmt
void WhileStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave,
                                    ThreadgroupContext &tg, int ourStackIndex,
                                    uint32_t &headerBlockId,
                                    uint32_t &mergeBlockId) {
  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Get current merge stack for block creation
  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);

  // Create loop blocks (header, merge) - pass current execution path
  auto [hBlockId, mBlockId] =
      tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                          currentMergeStack, lane.executionPath);
  headerBlockId = hBlockId;
  mergeBlockId = mBlockId;

  // Push merge point for loop divergence
  std::set<uint32_t> divergentBlocks = {headerBlockId};
  tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                    parentBlockId, divergentBlocks);

  // Push execution state onto stack
  LaneContext::BlockExecutionState newState(
      static_cast<const void *>(this),
      LaneContext::ControlFlowPhase::EvaluatingCondition, 0);
  newState.loopIteration = 0;
  newState.loopHeaderBlockId = headerBlockId;
  newState.loopMergeBlockId = mergeBlockId;

  lane.executionStack.push_back(newState);

  // Move to loop header block
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
}

// Helper method for setting up iteration-specific blocks in WhileStmt
void WhileStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg, int ourStackIndex,
                                     uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Check if we need an iteration-specific starting block
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique
  // iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      needsIterationBlock = true;
      INTERPRETER_DEBUG_LOG(
          "DEBUG: WhileStmt - Lane " << lane.laneId
          << " needs iteration block (first statement is not control flow)");
    } else {
      INTERPRETER_DEBUG_LOG(
          "DEBUG: WhileStmt - Lane " << lane.laneId
          << " no iteration block needed (first statement is control flow)");
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block
    iterationStartBlockId = lane.executionStack[ourStackIndex].loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x3000);

      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId, currentMergeStack,
          true, lane.executionPath);

      iterationStartBlockId = tg.findBlockByIdentity(iterationIdentity);

      if (iterationStartBlockId == 0) {
        std::map<WaveId, std::set<LaneId>> expectedLanes =
            tg.getCurrentBlockParticipants(headerBlockId);
        iterationStartBlockId =
            tg.findOrCreateBlockForPath(iterationIdentity, expectedLanes);
        INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                  << " created iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration);
      }

      lane.executionStack[ourStackIndex].loopBodyBlockId =
          iterationStartBlockId;
    }

    if (tg.getCurrentBlock(wave.waveId, lane.laneId) != iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId,
                                              wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                << " moved to iteration starting block "
                << iterationStartBlockId);
    }
  } else {
    // No iteration block needed - but we still need to ensure unique execution
    // context per iteration Only push merge point if we're at the beginning of
    // the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees
      // different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                << " merge stack size: " << currentMergeStack.size());
      for (size_t i = 0; i < currentMergeStack.size(); i++) {
        INTERPRETER_DEBUG_LOG("  Stack[" << i << "]: sourceStatement="
                  << currentMergeStack[i].sourceStatement);
      }
      INTERPRETER_DEBUG_LOG("  Looking for iterationMarker=" << iterationMarker);
      bool alreadyPushed =
          hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack");
      }

      if (!alreadyPushed) {
        std::set<uint32_t>
            emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId),
                          emptyDivergentBlocks);
        INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                  << " pushed iteration merge point " << iterationMarker
                  << " for iteration " << ourEntry.loopIteration);
      }

      INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                << " executing directly in current block "
                << tg.getCurrentBlock(wave.waveId, lane.laneId)
                << " (no iteration block needed, but merge stack modified)");
    }
  }
}

// Helper method for body completion cleanup in WhileStmt
void WhileStmt::cleanupAfterBodyExecution(LaneContext &lane, WaveContext &wave,
                                          ThreadgroupContext &tg,
                                          int ourStackIndex,
                                          uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Clean up iteration-specific merge point
  const void *iterationMarker =
      reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(this) +
                                     (ourEntry.loopIteration << 16) + 0x5000);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration " << ourEntry.loopIteration);
  }

  // Move back to header block for next iteration
  uint32_t finalBlock =
      tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " body completed in block " << finalBlock
            << ", moving to header block " << headerBlockId
            << " for next iteration");
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  ourEntry.phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  ourEntry.loopIteration++;
  ourEntry.statementIndex = 0;

  setThreadStateIfUnprotected(lane);
}

// Helper method for loop exit/reconverging phase in WhileStmt
void WhileStmt::handleLoopExit(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg, int ourStackIndex,
                               uint32_t mergeBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " reconverging from while loop to merge block " << mergeBlockId);

  // Debug: Check execution stack state before popping
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " execution stack size before pop: "
            << lane.executionStack.size());

  // Pop our entry from execution stack
  lane.executionStack.pop_back();

  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " execution stack size after pop: " << lane.executionStack.size());

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId,
                                          lane.laneId);

  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " successfully moved to merge block " << mergeBlockId
            << " and marked as Participating");

  // Loop has completed, restore active state and allow progression to next
  // statement
  lane.isActive = lane.isActive && !lane.hasReturned;

  // CRITICAL FIX: Set lane state to Ready to allow statement progression
  // This prevents infinite re-execution of the WhileStmt and allows
  // executeOneStep() to increment currentStatement
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::Ready;
  }

  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " completing reconvergence, set state to Ready for statement "
               "progression");
}

// Helper method for break exception handling in WhileStmt
void WhileStmt::handleBreakException(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg, int ourStackIndex,
                                     uint32_t headerBlockId) {
  // Break - exit loop
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " breaking from while loop");
  tg.popMergePoint(wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);

  setThreadStateIfUnprotected(lane);
}

// Helper method for continue exception handling in WhileStmt
void WhileStmt::handleContinueException(LaneContext &lane, WaveContext &wave,
                                        ThreadgroupContext &tg,
                                        int ourStackIndex,
                                        uint32_t headerBlockId) {
  // Continue - go to next iteration
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " continuing while loop");

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId,
                                        ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Marked lane " << lane.laneId
              << " as Left from block " << ourEntry.loopBodyBlockId);
  }

  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration++;

  // Clean up - remove from all nested blocks this lane is abandoning
  if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
    tg.removeThreadFromAllSets(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
    tg.removeThreadFromNestedBlocks(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
  }

  // Move lane back to header block for proper context
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  lane.executionStack[ourStackIndex].loopBodyBlockId =
      0; // Reset for next iteration
  tg.popMergePoint(wave.waveId, lane.laneId);

  // Set state to WaitingForResume to prevent currentStatement increment
  setThreadStateIfUnprotected(lane);
}

// Result-based versions of WhileStmt helper methods
Result<Unit, ExecutionError> WhileStmt::evaluateConditionPhase_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration
            << " (Result-based)");

  // Debug: Print all variables
  INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId << " variables: ");
  for (const auto &var : lane.variables) {
    INTERPRETER_DEBUG_LOG(var.first << "=" << var.second.toString() << " ");
  }

  // Check loop condition using Result-based evaluation
  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit,
                             ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all
    // iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId,
                               lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(
        headerBlockId, wave.waveId,
        lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  // Condition passed, move to body execution
  ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
  ourEntry.statementIndex = 0;
  setThreadStateIfUnprotected(lane);

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> WhileStmt::executeBodyStatements_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Execute statements using Result-based approach
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement =
        tg.getCurrentBlock(wave.waveId, lane.laneId);
    INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
              << " executing statement " << i << " in block "
              << blockBeforeStatement << " (Result-based)");

    // Use Result-based execute_with_error_handling for proper control flow
    // handling
    auto stmt_result = body_[i]->execute_with_error_handling(lane, wave, tg);
    if (stmt_result.is_err()) {
      return stmt_result; // Propagate the error
    }

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        INTERPRETER_DEBUG_LOG(
            "DEBUG: WhileStmt - Lane " << lane.laneId
            << " popped iteration merge point on early return (Result-based)"
        );
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement =
          tg.getCurrentBlock(wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                << " child statement needs resume (Result-based)");
      INTERPRETER_DEBUG_LOG("  Block before: " << blockBeforeStatement
                << ", Block after: " << blockAfterStatement);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      INTERPRETER_DEBUG_LOG("DEBUG: WhileStmt - Lane " << lane.laneId
                << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement "
                << i << " (likely merge block, Result-based)");
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

// DoWhileStmt implementation
DoWhileStmt::DoWhileStmt(std::vector<std::unique_ptr<Statement>> body,
                         std::unique_ptr<Expression> cond)
    : body_(std::move(body)), condition_(std::move(cond)) {}

// Helper methods for DoWhileStmt execute phases
void DoWhileStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg, int ourStackIndex,
                                      uint32_t &headerBlockId,
                                      uint32_t &mergeBlockId) {
  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Get current merge stack for block creation
  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);

  // Create loop blocks (header, merge) - pass current execution path
  // For do-while, header is where condition is checked
  auto [header, merge] =
      tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                          currentMergeStack, lane.executionPath);
  headerBlockId = header;
  mergeBlockId = merge;

  // Push merge point for loop divergence
  std::set<uint32_t> divergentBlocks = {headerBlockId};
  tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this),
                    parentBlockId, divergentBlocks);

  // Push execution state onto stack - DoWhile starts with body execution
  LaneContext::BlockExecutionState newState(
      static_cast<const void *>(this),
      LaneContext::ControlFlowPhase::ExecutingBody, 0);
  newState.loopIteration = 0;
  newState.loopHeaderBlockId = headerBlockId;
  newState.loopMergeBlockId = mergeBlockId;

  lane.executionStack.push_back(newState);
}

void DoWhileStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg,
                                       int ourStackIndex,
                                       uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Check if we need an iteration-specific starting block
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique
  // iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      needsIterationBlock = true;
INTERPRETER_DEBUG_LOG( "DEBUG: DoWhileStmt - Lane " << lane.laneId
          << " needs iteration block (first statement is not control flow)");
    } else {
      INTERPRETER_DEBUG_LOG(
          "DEBUG: DoWhileStmt - Lane " << lane.laneId
          << " no iteration block needed (first statement is control flow)"
      );
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block
    iterationStartBlockId = ourEntry.loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x3000);

      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId, currentMergeStack,
          true, lane.executionPath);

      // Try to find existing block first
      iterationStartBlockId = tg.findBlockByIdentity(iterationIdentity);

      if (iterationStartBlockId == 0) {
        // Create new block only if none exists
        std::map<WaveId, std::set<LaneId>> expectedLanes;
        if (ourEntry.loopIteration == 0) {
          uint32_t parentBlockId = iterationIdentity.parentBlockId;
          expectedLanes = tg.getCurrentBlockParticipants(parentBlockId);
        } else {
          expectedLanes = tg.getCurrentBlockParticipants(headerBlockId);
        }
        iterationStartBlockId =
            tg.findOrCreateBlockForPath(iterationIdentity, expectedLanes);
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " created iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration);
      } else {
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " found existing iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration);
      }

      ourEntry.loopBodyBlockId = iterationStartBlockId;
    }

    // Move to iteration starting block if not already there
    if (tg.getCurrentBlock(wave.waveId, lane.laneId) != iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId,
                                              wave.waveId, lane.laneId);
      INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " moved to iteration starting block "
                << iterationStartBlockId << " for iteration "
                << ourEntry.loopIteration);
    }
  } else {
    // No iteration block needed - but we still need to ensure unique
    // execution context per iteration Only push merge point if we're at the
    // beginning of the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees
      // different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      bool alreadyPushed =
          hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack"
        );
      }

      if (!alreadyPushed) {
        std::set<uint32_t>
            emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId),
                          emptyDivergentBlocks);
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " pushed iteration merge point " << iterationMarker
                  << " for iteration " << ourEntry.loopIteration);
      }

      INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " executing directly in current block "
                << tg.getCurrentBlock(wave.waveId, lane.laneId)
                << " (no iteration block needed, but merge stack modified)"
      );
    }
  }
}

void DoWhileStmt::cleanupAfterBodyExecution(LaneContext &lane,
                                            WaveContext &wave,
                                            ThreadgroupContext &tg,
                                            int ourStackIndex,
                                            uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Body completed - clean up iteration-specific merge point
  const void *iterationMarker =
      reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(this) +
                                     (ourEntry.loopIteration << 16) + 0x5000);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration "
              << lane.executionStack[ourStackIndex].loopIteration);
  }

  // Completed body execution - move back to header block and to condition
  // evaluation
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " completed body for iteration "
            << lane.executionStack[ourStackIndex].loopIteration);
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;
}

void DoWhileStmt::handleLoopExit(LaneContext &lane, WaveContext &wave,
                                 ThreadgroupContext &tg, int ourStackIndex,
                                 uint32_t mergeBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " reconverging from do-while loop");

  // Pop our entry from execution stack
  lane.executionStack.pop_back();

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  // tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId,
                                          lane.laneId);

  // Loop has completed, restore active state
  lane.isActive = lane.isActive && !lane.hasReturned;
}

void DoWhileStmt::handleBreakException(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg,
                                       int ourStackIndex,
                                       uint32_t headerBlockId,
                                       uint32_t mergeBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " breaking from do-while loop");
  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);

  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
  setThreadStateIfUnprotected(lane);
}

void DoWhileStmt::handleContinueException(LaneContext &lane, WaveContext &wave,
                                          ThreadgroupContext &tg,
                                          int ourStackIndex,
                                          uint32_t headerBlockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " continuing do-while loop");

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId,
                                        ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Marked lane " << lane.laneId
              << " as Left from block " << ourEntry.loopBodyBlockId);
  }

  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;

  // Clean up - remove from all nested blocks this lane is abandoning
  if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
    tg.removeThreadFromAllSets(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
    tg.removeThreadFromNestedBlocks(
        lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
        lane.laneId);
  }

  // Move lane back to header block for proper context
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  lane.executionStack[ourStackIndex].loopBodyBlockId =
      0; // Reset for next iteration

  // Set state to WaitingForResume to prevent currentStatement increment
  setThreadStateIfUnprotected(lane);
}

// Phase-based Result methods for DoWhileStmt
Result<Unit, ExecutionError> DoWhileStmt::executeBody(LaneContext &lane,
                                                      WaveContext &wave,
                                                      ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " executing body (Result-based)");

  for (const auto &stmt : body_) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      INTERPRETER_DEBUG_LOG("DoWhileStmt - Body encountered error");

      // Break and continue are normal control flow for do-while loops
      if (error == ExecutionError::ControlFlowBreak) {
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Break encountered in body");
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Continue encountered in body");
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowContinue);
      }

      // Other errors propagate up
      return result;
    }
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<bool, ExecutionError>
DoWhileStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)");

  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<bool, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    auto condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), bool, ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Condition result: " << shouldContinue);
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError>
DoWhileStmt::execute_result(LaneContext &lane, WaveContext &wave,
                            ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting")
            << " Result-based do-while loop");

  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;

  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId,
                        mergeBlockId);
    ourStackIndex = lane.executionStack.size() - 1;
  }

  // Get our execution state
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (isResuming) {
    headerBlockId = ourEntry.loopHeaderBlockId;
    mergeBlockId = ourEntry.loopMergeBlockId;
  }

  // Result-based state machine for do-while loop execution
  switch (ourEntry.phase) {
  case LaneContext::ControlFlowPhase::ExecutingBody: {
    INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
              << " executing body for iteration " << ourEntry.loopIteration
              << " from statement " << ourEntry.statementIndex
              << " (Result-based)");

    // Only set up iteration blocks if we're starting the body (statement index 0)
    if (ourEntry.statementIndex == 0) {
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);
    }

    // Execute body statements using Result-based helper method
    #ifdef _MSC_VER
      auto _result = executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(executeBodyStatements_result(lane, wave, tg, ourStackIndex,
                                            headerBlockId),
               Unit, ExecutionError);
    #endif

    if (lane.hasReturned || lane.state != ThreadState::Ready) {
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Clean up after body execution using non-throwing helper method
    cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::EvaluatingCondition: {
    // Evaluate condition using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
      TRY_RESULT(evaluateConditionPhase_result(lane, wave, tg, ourStackIndex,
                                             headerBlockId),
               Unit, ExecutionError);
    #endif

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::Reconverging: {
    // Handle loop exit using non-throwing helper method
    handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  default:
    INTERPRETER_DEBUG_LOG("ERROR: DoWhileStmt - Lane " << lane.laneId
              << " unexpected phase in Result-based execution");
    lane.executionStack.pop_back();
    return Ok<Unit, ExecutionError>(Unit{});
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

// Specialized wrapper function for DoWhileStmt-specific error handling
Result<Unit, ExecutionError>
DoWhileStmt::execute_with_error_handling(LaneContext &lane, WaveContext &wave,
                                         ThreadgroupContext &tg) {
  auto result = execute_result(lane, wave, tg);
  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();

    switch (error) {
    case ExecutionError::WaveOperationWait:
      // DoWhileStmt-specific: Wave operation already set the lane to
      // WaitingForWave Just propagate the error without changing state
      return result; // Propagate for parent to handle

    case ExecutionError::ControlFlowBreak:
      // DoWhileStmt consumes break: exit loop cleanly
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          uint32_t mergeBlockId =
              lane.executionStack[ourStackIndex].loopMergeBlockId;
          handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId,
                               mergeBlockId);
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - break handled

    case ExecutionError::ControlFlowContinue:
      // DoWhileStmt consumes continue: jump to condition phase
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          uint32_t headerBlockId =
              lane.executionStack[ourStackIndex].loopHeaderBlockId;
          handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - continue handled

    default:
      return result; // Propagate unknown errors
    }
  }
  return result; // Success case
}

// Result-based versions of DoWhileStmt helper methods
Result<Unit, ExecutionError> DoWhileStmt::executeBodyStatements_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Execute statements from where we left off using Result-based approach
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    // Use Result-based execute_with_error_handling for proper control flow
    // handling
    auto stmt_result = body_[i]->execute_with_error_handling(lane, wave, tg);
    if (stmt_result.is_err()) {
      return stmt_result; // Propagate the error
    }

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
          0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        INTERPRETER_DEBUG_LOG(
            "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " popped iteration merge point on early return (Result-based)"
        );
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " child statement needs resume (Result-based)");
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> DoWhileStmt::evaluateConditionPhase_result(
    LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " evaluating condition after iteration "
            << ourEntry.loopIteration << " (Result-based)");

  // Check loop condition using Result-based evaluation
  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit,
                             ExecutionError);
  #endif
  bool shouldContinue = condVal.asBool();
    if (tg.interpreter) {
      tg.interpreter->onControlFlow(lane, wave, tg, this, shouldContinue);
    }
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all
    // iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId,
                               lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(
        headerBlockId, wave.waveId,
        lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    lane.executionStack[ourStackIndex].phase =
        LaneContext::ControlFlowPhase::Reconverging;
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  // Condition passed, move to next iteration body execution
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  lane.executionStack[ourStackIndex].loopIteration++;
  setThreadStateIfUnprotected(lane);

  return Ok<Unit, ExecutionError>(Unit{});
}

// SwitchStmt implementation
SwitchStmt::SwitchStmt(std::unique_ptr<Expression> cond)
    : condition_(std::move(cond)) {}

void SwitchStmt::addCase(int value,
                         std::vector<std::unique_ptr<Statement>> stmts) {
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt::addCase - adding case " << value);
  cases_.push_back({value, std::move(stmts)});
}

void SwitchStmt::addDefault(std::vector<std::unique_ptr<Statement>> stmts) {
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt::addDefault - adding default case");
  cases_.push_back({std::nullopt, std::move(stmts)});
}

// Helper methods for SwitchStmt execute phases
void SwitchStmt::setupSwitchExecution(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg,
                                      int ourStackIndex) {
  // Create blocks for all cases
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);

  // Extract case values and check for default
  std::vector<int> caseValues;
  bool hasDefault = false;
  for (const auto &caseBlock : cases_) {
    if (caseBlock.value.has_value()) {
      caseValues.push_back(caseBlock.value.value());
    } else {
      hasDefault = true;
    }
  }

  // Create switch blocks and store in execution stack
  auto &ourEntry = lane.executionStack[ourStackIndex];
  auto allBlockIds = tg.createSwitchCaseBlocks(
      static_cast<const void *>(this), parentBlockId, currentMergeStack,
      caseValues, hasDefault, lane.executionPath);

  // Extract blocks: [headerBlockId, caseBlock1, caseBlock2, ..., mergeBlockId]
  if (allBlockIds.size() < 2) {
    INTERPRETER_DEBUG_LOG("ERROR: SwitchStmt - Insufficient blocks created");
    return;
  }

  ourEntry.switchHeaderBlockId =
      allBlockIds[0]; // First element is header block
  ourEntry.switchMergeBlockId =
      allBlockIds.back(); // Last element is merge block

  // Extract case blocks (everything between header and merge)
  ourEntry.switchCaseBlockIds =
      std::vector<uint32_t>(allBlockIds.begin() + 1, allBlockIds.end() - 1);

  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Created header block "
            << ourEntry.switchHeaderBlockId << ", "
            << ourEntry.switchCaseBlockIds.size()
            << " case blocks, and merge block " << ourEntry.switchMergeBlockId
  );

  // Move to header block for condition evaluation
  tg.moveThreadFromUnknownToParticipating(ourEntry.switchHeaderBlockId,
                                          wave.waveId, lane.laneId);
}

void SwitchStmt::findMatchingCase(LaneContext &lane, WaveContext &wave,
                                  ThreadgroupContext &tg, int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Find which case this lane should execute
  int switchValue = ourEntry.switchValue.asInt();
  size_t matchingCaseIndex = SIZE_MAX;

  for (size_t i = 0; i < cases_.size(); ++i) {
    if (cases_[i].value.has_value() && cases_[i].value.value() == switchValue) {
      matchingCaseIndex = i;
      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " matched case " << cases_[i].value.value() << " at index "
                << i);
      break;
    } else if (!cases_[i].value.has_value()) {
      // Default case - only use if no exact match found
      if (matchingCaseIndex == SIZE_MAX) {
        matchingCaseIndex = i;
        INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " matched default case at index " << i);
      }
    }
  }

  if (matchingCaseIndex == SIZE_MAX) {
    // No matching case - remove from all case blocks and exit switch via
    // reconvergence
    INTERPRETER_DEBUG_LOG(
        "DEBUG: SwitchStmt - Lane " << lane.laneId
        << " no matching case found, cleaning up and entering reconvergence"
    );

    // Remove this lane from ALL case blocks (it will never execute any cases)
    for (size_t i = 0; i < ourEntry.switchCaseBlockIds.size(); ++i) {
      uint32_t caseBlockId = ourEntry.switchCaseBlockIds[i];
      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " removing from case block " << caseBlockId << " (case " << i
                << ") - no matching case");
      tg.removeThreadFromAllSets(caseBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(caseBlockId, wave.waveId, lane.laneId);
    }

    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    setThreadStateIfUnprotected(lane);
    return;
  }

  // Set up for case execution and move to appropriate block
  ourEntry.caseIndex = matchingCaseIndex;
  ourEntry.statementIndex = 0;
  ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingCase;

  // Move lane to the appropriate case block and remove from previous cases
  if (matchingCaseIndex < ourEntry.switchCaseBlockIds.size()) {
    uint32_t chosenCaseBlockId = ourEntry.switchCaseBlockIds[matchingCaseIndex];
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " moving to case block " << chosenCaseBlockId << " for case "
              << matchingCaseIndex);

    // Move to the first matching case block only (like if/loop pattern)
    uint32_t firstCaseBlockId = ourEntry.switchCaseBlockIds[matchingCaseIndex];
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " moving from header to first case block " << firstCaseBlockId
              << " (case " << matchingCaseIndex << ")");

    // tg.assignLaneToBlock(wave.waveId, lane.laneId, firstCaseBlockId);
    tg.moveThreadFromUnknownToParticipating(firstCaseBlockId, wave.waveId,
                                            lane.laneId);

    // Remove from all previous case blocks (cases this lane will never execute)
    for (size_t i = 0; i < matchingCaseIndex; ++i) {
      uint32_t previousCaseBlockId = ourEntry.switchCaseBlockIds[i];
      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " removing from previous case block " << previousCaseBlockId
                << " (case " << i << ")");
      tg.removeThreadFromUnknown(previousCaseBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(previousCaseBlockId, wave.waveId,
                                      lane.laneId);
    }

    // Remove from header block
    tg.removeThreadFromAllSets(ourEntry.switchHeaderBlockId, wave.waveId,
                               lane.laneId);

    // Transition to executing the case
    ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingCase;
  }
}

void SwitchStmt::handleReconvergence(LaneContext &lane, WaveContext &wave,
                                     ThreadgroupContext &tg,
                                     int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
            << " reconverging from switch");

  // Get merge block before popping execution stack
  uint32_t mergeBlockId = ourEntry.switchMergeBlockId;

  // Pop our entry from execution stack
  lane.executionStack.pop_back();

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  // tg.assignLaneToBlock(wave.waveId, lane.laneId, mergeBlockId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId,
                                          lane.laneId);
}

void SwitchStmt::handleBreakException(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg,
                                      int ourStackIndex) {
  // Break - exit switch and clean up subsequent case blocks
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
            << " breaking from switch");

  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Remove this lane from ALL case blocks (it's breaking out of the entire
  // switch)
  for (size_t i = 0; i < ourEntry.switchCaseBlockIds.size(); ++i) {
    uint32_t caseBlockId = ourEntry.switchCaseBlockIds[i];
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " removing from case block " << caseBlockId << " (case " << i
              << ") due to break");
    tg.removeThreadFromAllSets(caseBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(caseBlockId, wave.waveId, lane.laneId);
  }
  // Use Reconverging phase instead of direct exit
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
}

// Phase-based Result methods for SwitchStmt
Result<int, ExecutionError>
SwitchStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                              ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)");

  #ifdef _MSC_VER
    auto _result = condition_->evaluate_result(lane, wave, tg);
    if (_result.is_err()) return Err<int, ExecutionError>(_result.unwrap_err());
    Value condVal = _result.unwrap();
  #else
    auto condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), int, ExecutionError);
  #endif
  int switchValue = condVal.asInt();

  INTERPRETER_DEBUG_LOG("SwitchStmt - Switch value: "<< switchValue);
  return Ok<int, ExecutionError>(switchValue);
}

Result<Unit, ExecutionError> SwitchStmt::executeCase(size_t caseIndex,
                                                     LaneContext &lane,
                                                     WaveContext &wave,
                                                     ThreadgroupContext &tg) {
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId << " executing case "
            << caseIndex << " (Result-based)");

  const auto &caseBlock = cases_[caseIndex];

  for (const auto &stmt : caseBlock.statements) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      INTERPRETER_DEBUG_LOG("SwitchStmt - Case encountered error");

      // Break exits the switch (normal behavior)
      if (error == ExecutionError::ControlFlowBreak) {
        INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Break encountered in case");
        return Ok<Unit, ExecutionError>(
            Unit{}); // Break stops switch execution successfully
      }

      // Continue should be handled by containing loop, not switch
      // Other errors propagate up
      return result;
    }
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
SwitchStmt::execute_result(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting")
            << " Result-based switch statement");

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " starting fresh execution (pushed to stack depth="
              << lane.executionStack.size() << ", this=" << this << ")"
    );

    // Setup switch execution blocks using non-throwing helper method
    setupSwitchExecution(lane, wave, tg, ourStackIndex);
  } else {
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " resuming execution (found at stack index=" << ourStackIndex
              << ", current stack depth=" << lane.executionStack.size()
              << ", this=" << this << ")");
  }

  auto &ourEntry = lane.executionStack[ourStackIndex];
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId << " in phase "
            << LaneContext::getPhaseString(ourEntry.phase)
            << " (Result-based, stack depth=" << lane.executionStack.size()
            << ", our index=" << ourStackIndex << ", this=" << this << ")"
  );

  // Result-based state machine for switch statement execution
  switch (ourEntry.phase) {
  case LaneContext::ControlFlowPhase::EvaluatingCondition: {
    // Evaluate switch value using Result-based helper method
    #ifdef _MSC_VER
      auto _result = evaluateSwitchValue_result(lane, wave, tg, ourStackIndex);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
      TRY_RESULT(evaluateSwitchValue_result(lane, wave, tg, ourStackIndex), Unit,
               ExecutionError);
    #endif

    // Find matching case and set up execution using non-throwing helper method
    findMatchingCase(lane, wave, tg, ourStackIndex);

    // Check if we're moving to reconverging (no matching case)
    if (ourEntry.phase == LaneContext::ControlFlowPhase::Reconverging) {
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement
                                             // increment, will resume later
  }

  case LaneContext::ControlFlowPhase::ExecutingCase: {
    // Execute case statements using Result-based helper method
    #ifdef _MSC_VER
      auto _result = executeCaseStatements_result(lane, wave, tg, ourStackIndex);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
    #else
    TRY_RESULT(executeCaseStatements_result(lane, wave, tg, ourStackIndex),
               Unit, ExecutionError);
    #endif

    if (lane.hasReturned || lane.state != ThreadState::Ready) {
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Set state to WaitingForResume to prevent currentStatement increment
    setThreadStateIfUnprotected(lane);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  case LaneContext::ControlFlowPhase::Reconverging: {
    // Handle reconvergence using non-throwing helper method
    handleReconvergence(lane, wave, tg, ourStackIndex);
    return Ok<Unit, ExecutionError>(Unit{});
  }

  default:
    INTERPRETER_DEBUG_LOG("ERROR: SwitchStmt - Lane " << lane.laneId
              << " unexpected phase in Result-based execution");
    lane.executionStack.pop_back();
    return Ok<Unit, ExecutionError>(Unit{});
  }
}

// Result-based helper functions for SwitchStmt
Result<Unit, ExecutionError>
SwitchStmt::evaluateSwitchValue_result(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg,
                                       int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
            << " evaluating switch condition (Result-based)");

  // Only evaluate condition if not already evaluated
  if (!ourEntry.conditionEvaluated) {
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " evaluating condition for first time");

    #ifdef _MSC_VER
      auto _result = condition_->evaluate_result(lane, wave, tg);
      if (_result.is_err()) return Err<Unit, ExecutionError>(_result.unwrap_err());
      Value condValue = _result.unwrap();
    #else
      auto condValue = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
    #endif

    lane.executionStack[ourStackIndex].switchValue = condValue;
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " switch condition evaluated to: "
              << condValue.asInt());
  } else {
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " using cached condition result="
              << ourEntry.switchValue.asInt());
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError>
SwitchStmt::executeCaseStatements_result(LaneContext &lane, WaveContext &wave,
                                         ThreadgroupContext &tg,
                                         int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Execute all statements from current position until case/switch completion
  while (ourEntry.caseIndex < cases_.size()) {
    const auto &caseBlock = cases_[ourEntry.caseIndex];
    std::string caseLabel = caseBlock.value.has_value()
                                ? std::to_string(caseBlock.value.value())
                                : "default";

    // Execute all statements in current case from saved position
    for (size_t i = ourEntry.statementIndex; i < caseBlock.statements.size();
         i++) {
      lane.executionStack[ourStackIndex].statementIndex = i;

      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " executing statement " << i << " in case " << caseLabel
                << " (Result-based)");

      // Use Result-based execute_result instead of exception-based execute
      // Match exception-based semantics by calling execute_result directly
      auto stmt_result =
          caseBlock.statements[i]->execute_result(lane, wave, tg);
      if (stmt_result.is_err()) {
        return stmt_result; // Propagate the error
      }

      if (lane.hasReturned) {
        INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " popping stack due to return (depth "
                  << lane.executionStack.size() << "->"
                  << (lane.executionStack.size() - 1) << ", this=" << this
                  << ")");
        // Clean up and return
        lane.executionStack.pop_back();
        tg.popMergePoint(wave.waveId, lane.laneId);
        return Ok<Unit, ExecutionError>(Unit{});
      }

      if (lane.state != ThreadState::Ready) {
        // Child statement needs to resume - don't continue
        INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " child statement not ready (state=" << (int)lane.state
                  << "), not advancing statementIndex=" << i);
        return Ok<Unit, ExecutionError>(Unit{});
      }
      lane.executionStack[ourStackIndex].statementIndex = i + 1;
    }

    // Check if lane is ready before fallthrough - if waiting for wave
    // operations, don't move to next case
    if (lane.state != ThreadState::Ready) {
      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " completed case " << caseLabel
                << " but is not Ready (state=" << (int)lane.state
                << "), pausing before fallthrough");
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Current case completed - move to next case (fallthrough)
    INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
              << " completed case " << caseLabel << ", falling through to next"
    );

    size_t nextCaseIndex = ourEntry.caseIndex + 1;

    // Move lane to next case block if it exists
    if (nextCaseIndex < ourEntry.switchCaseBlockIds.size()) {
      uint32_t currentCaseBlockId =
          ourEntry.switchCaseBlockIds[ourEntry.caseIndex];
      uint32_t nextCaseBlockId = ourEntry.switchCaseBlockIds[nextCaseIndex];

      INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
                << " moving from case block " << currentCaseBlockId
                << " to case block " << nextCaseBlockId << " (fallthrough)"
      );

      // Move to next case block (fallthrough)
      tg.moveThreadFromUnknownToParticipating(nextCaseBlockId, wave.waveId,
                                              lane.laneId);

      // Remove from current case block
      tg.removeThreadFromAllSets(currentCaseBlockId, wave.waveId, lane.laneId);
    }

    // Move to next case
    lane.executionStack[ourStackIndex].caseIndex++;
    lane.executionStack[ourStackIndex].statementIndex = 0;
  }

  // All cases completed - enter reconvergence
  INTERPRETER_DEBUG_LOG("DEBUG: SwitchStmt - Lane " << lane.laneId
            << " completed all cases, entering reconvergence");
  ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;

  return Ok<Unit, ExecutionError>(Unit{});
}

std::string SwitchStmt::toString() const {
  std::string result = "switch (" + condition_->toString() + ") {\n";
  for (const auto &caseBlock : cases_) {
    if (caseBlock.value.has_value()) {
      result += "  case " + std::to_string(caseBlock.value.value()) + ": {\n";
    } else {
      result += "  default: {\n";
    }
    for (const auto &stmt : caseBlock.statements) {
      result += "    " + stmt->toString() + "\n";
    }
    result += "  }\n";  // Close the case scope
  }
  result += "}";
  return result;
}

// Specialized wrapper function for SwitchStmt-specific error handling
Result<Unit, ExecutionError>
SwitchStmt::execute_with_error_handling(LaneContext &lane, WaveContext &wave,
                                        ThreadgroupContext &tg) {
  auto result = execute_result(lane, wave, tg);
  if (result.is_err()) {
    ExecutionError error = result.unwrap_err();

    switch (error) {
    case ExecutionError::WaveOperationWait:
      // SwitchStmt-specific: Wave operation already set the lane to
      // WaitingForWave Just propagate the error without changing state
      return result; // Propagate for parent to handle

    case ExecutionError::ControlFlowBreak:
      // SwitchStmt consumes break: exit switch cleanly
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          handleBreakException(lane, wave, tg, ourStackIndex);
          setThreadStateIfUnprotected(lane);

          // Restore active state (reconvergence)
          lane.isActive = lane.isActive && !lane.hasReturned;
        }
      }
      return Ok<Unit, ExecutionError>(Unit{}); // Success - break handled

    case ExecutionError::ControlFlowContinue:
      // SwitchStmt does NOT consume continue - propagate to enclosing loop
      // Clean up our stack entry
      {
        int ourStackIndex = findStackIndex(lane);
        if (ourStackIndex >= 0) {
          lane.executionStack.pop_back();
        }
      }
      return result; // Propagate to parent loop

    default:
      return result; // Propagate unknown errors
    }
  }
  return result; // Success case
}

Result<Unit, ExecutionError>
ContinueStmt::execute_result(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based continue - no exceptions thrown!
  INTERPRETER_DEBUG_LOG("DEBUG: ContinueStmt - Lane " << lane.laneId
            << " executing continue via Result");
  return Err<Unit, ExecutionError>(ExecutionError::ControlFlowContinue);
}

Result<Unit, ExecutionError> BreakStmt::execute_result(LaneContext &lane,
                                                       WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based break - no exceptions thrown!
  INTERPRETER_DEBUG_LOG("DEBUG: BreakStmt - Lane " << lane.laneId
            << " executing break via Result");

  // Debug: print execution stack to see nesting
  INTERPRETER_DEBUG_LOG("DEBUG: BreakStmt - Execution stack size: "
            << lane.executionStack.size());
  for (size_t i = 0; i < lane.executionStack.size(); i++) {
    INTERPRETER_DEBUG_LOG("  Stack[" << i
              << "]: statement=" << lane.executionStack[i].statement
              << ", phase=" << static_cast<int>(lane.executionStack[i].phase));
  }

  return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
}

void ThreadgroupContext::assignLaneToBlock(WaveId waveId, LaneId laneId,
                                           uint32_t blockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: assignLaneToBlock - START: lane " << laneId
            << " being assigned to block " << blockId);

  // Show current state before assignment
  auto partBefore = membershipRegistry.getParticipatingLanes(waveId, blockId);
  auto waitBefore = membershipRegistry.getWaitingLanes(waveId, blockId);
  auto unknownBefore = membershipRegistry.getUnknownLanes(waveId, blockId);
  // INTERPRETER_DEBUG_LOG("DEBUG: assignLaneToBlock - BEFORE: block " << blockId << " has "
  //           << partBefore.size() << " participating lanes");
  // if (!partBefore.empty()) {
  //   std::cout << " (";
  //   for (auto it = partBefore.begin(); it != partBefore.end(); ++it) {
  //     if (it != partBefore.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << ", " << waitBefore.size() << " waiting lanes";
  // if (!waitBefore.empty()) {
  //   std::cout << " (";
  //   for (auto it = waitBefore.begin(); it != waitBefore.end(); ++it) {
  //     if (it != waitBefore.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << ", " << unknownBefore.size() << " unknown lanes";
  // if (!unknownBefore.empty()) {
  //   std::cout << " (";
  //   for (auto it = unknownBefore.begin(); it != unknownBefore.end(); ++it) {
  //     if (it != unknownBefore.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << std::endl;

  // Don't move lanes that are waiting for wave operations or barriers
  if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
    auto state = waves[waveId]->lanes[laneId]->state;
    if (state == ThreadState::WaitingForWave ||
        state == ThreadState::WaitingAtBarrier) {
      INTERPRETER_DEBUG_LOG("DEBUG: Preventing move of waiting lane " << laneId
                << " (state=" << (int)state << ") to block " << blockId
      );
      return;
    }
  }

  // Check if lane is already in the target block
  uint32_t currentBlockId = membershipRegistry.getCurrentBlock(waveId, laneId);
  if (currentBlockId == blockId) {
    INTERPRETER_DEBUG_LOG("DEBUG: assignLaneToBlock - lane " << laneId
              << " already in block " << blockId
              << ", skipping redundant assignment");
    return;
  }

  // Remove lane from its current block if it's in one
  if (currentBlockId != 0) {
    auto *oldBlock = getBlock(currentBlockId);
    if (oldBlock) {
      // NEW: Don't remove from arrivedLanes if moving from header to iteration
      // block This keeps lanes in header's arrivedLanes until they exit the
      // loop entirely
      const auto *newBlock = getBlock(blockId);
      bool isHeaderToLoopBody =
          (oldBlock->getBlockType() == BlockType::LOOP_HEADER && newBlock &&
           newBlock->getBlockType() != BlockType::LOOP_EXIT);

      INTERPRETER_DEBUG_LOG("DEBUG: assignLaneToBlock - moving lane " << laneId
                << " from block " << currentBlockId << " (type "
                << (int)oldBlock->getBlockType() << ") to block " << blockId
                << " (type "
                << (int)(newBlock ? newBlock->getBlockType()
                                  : BlockType::REGULAR)
                << "), isHeaderToLoopBody=" << isHeaderToLoopBody);

      if (!isHeaderToLoopBody) {
        // Remove lane from old block
        membershipRegistry.setLaneStatus(waveId, laneId, currentBlockId,
                                         LaneBlockStatus::Left);
        INTERPRETER_DEBUG_LOG("DEBUG: Removed lane " << laneId << " from block "
                  << currentBlockId);
      } else {
        // Keep lane as Participating in header block for proper unknown lane
        // tracking This allows nested blocks to correctly determine expected
        // participants
        INTERPRETER_DEBUG_LOG("DEBUG: Keeping lane " << laneId
                  << " as Participating in header block " << currentBlockId
                  << " while also adding to loop body block " << blockId
        );
      }
    }
  }

  // Assign lane to new block in the registry (single source of truth)
  membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                   LaneBlockStatus::Participating);

  // Show state after assignment
  auto partAfter = membershipRegistry.getParticipatingLanes(waveId, blockId);
  auto waitAfter = membershipRegistry.getWaitingLanes(waveId, blockId);
  auto unknownAfter = membershipRegistry.getUnknownLanes(waveId, blockId);
  // std::cout << "DEBUG: assignLaneToBlock - AFTER: block " << blockId << " has "
  //           << partAfter.size() << " participating lanes";
  // if (!partAfter.empty()) {
  //   std::cout << " (";
  //   for (auto it = partAfter.begin(); it != partAfter.end(); ++it) {
  //     if (it != partAfter.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << ", " << waitAfter.size() << " waiting lanes";
  // if (!waitAfter.empty()) {
  //   std::cout << " (";
  //   for (auto it = waitAfter.begin(); it != waitAfter.end(); ++it) {
  //     if (it != waitAfter.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << ", " << unknownAfter.size() << " unknown lanes";
  // if (!unknownAfter.empty()) {
  //   std::cout << " (";
  //   for (auto it = unknownAfter.begin(); it != unknownAfter.end(); ++it) {
  //     if (it != unknownAfter.begin())
  //       std::cout << " ";
  //     std::cout << *it;
  //   }
  //   std::cout << ")";
  // }
  // std::cout << std::endl;

  INTERPRETER_DEBUG_LOG("DEBUG: assignLaneToBlock - END: lane " << laneId
            << " successfully assigned to block " << blockId);

  // Call hook for trace capture
  if (interpreter && waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
    interpreter->onLaneEnterBlock(*waves[waveId]->lanes[laneId], *waves[waveId], *this, blockId);
  }

  membershipRegistry.getCurrentBlock(waveId, laneId);
}

uint32_t ThreadgroupContext::getCurrentBlock(WaveId waveId,
                                             LaneId laneId) const {
  // Use membershipRegistry as the single source of truth
  uint32_t resultBlockId = membershipRegistry.getCurrentBlock(waveId, laneId);

  // DEBUG: Check if lane exists in multiple blocks (which shouldn't happen)
  std::vector<uint32_t> foundInBlocks;
  for (const auto &[blockId, block] : executionBlocks) {
    auto participatingLanes =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    auto waitingLanes = membershipRegistry.getWaitingLanes(waveId, blockId);

    bool isParticipating =
        participatingLanes.find(laneId) != participatingLanes.end();
    bool isWaiting = waitingLanes.find(laneId) != waitingLanes.end();

    if (isParticipating || isWaiting) {
      foundInBlocks.push_back(blockId);
    }
  }

  if (foundInBlocks.size() > 1) {
    INTERPRETER_DEBUG_LOG("WARNING: getCurrentBlock - Lane " << laneId
              << " found in multiple blocks: ");
    for (size_t i = 0; i < foundInBlocks.size(); ++i) {
      if (i > 0)
        INTERPRETER_DEBUG_LOG(", ");
      INTERPRETER_DEBUG_LOG(foundInBlocks[i]);
    }
    INTERPRETER_DEBUG_LOG(" (registry returned: " << resultBlockId << ")");
  }

  return resultBlockId;
}

void ThreadgroupContext::mergeExecutionPaths(
    const std::vector<uint32_t> &blockIds, uint32_t targetBlockId) {
  // Create or update the target block
  std::map<WaveId, std::set<LaneId>> mergedLanes;

  for (uint32_t blockId : blockIds) {
    const auto *block = getBlock(blockId);
    if (block) {
      // Merge lanes from this block, organized by wave
      for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
        auto participatingLanes =
            membershipRegistry.getParticipatingLanes(waveId, blockId);
        mergedLanes[waveId].insert(participatingLanes.begin(),
                                   participatingLanes.end());
      }
    }
  }

  // Count total lanes across all waves
  size_t totalMergedLanes = 0;
  for (const auto &[waveId, laneSet] : mergedLanes) {
    totalMergedLanes += laneSet.size();
  }

  // Create the target block with merged lanes
  if (!getBlock(targetBlockId)) {
    DynamicExecutionBlock targetBlock;
    targetBlock.setBlockId(targetBlockId);
    for (const auto &[waveId, laneSet] : mergedLanes) {
      for (LaneId laneId : laneSet) {
        membershipRegistry.setLaneStatus(waveId, laneId, targetBlockId,
                                         LaneBlockStatus::Participating);
        // targetBlock.addParticipatingLane(waveId, laneId);
      }
    }
    targetBlock.setProgramPoint(0);
    targetBlock.setIsConverged(totalMergedLanes == threadgroupSize);
    executionBlocks[targetBlockId] = targetBlock;
  } else {
    // Update existing target block
    auto &targetBlock = executionBlocks[targetBlockId];
    // Clear existing lanes and add merged ones (using registry)
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto lanes =
          membershipRegistry.getParticipatingLanes(waveId, targetBlockId);
      for (LaneId laneId : lanes) {
        membershipRegistry.setLaneStatus(waveId, laneId, targetBlockId,
                                         LaneBlockStatus::Left);
        // targetBlock.removeParticipatingLane(waveId, laneId);
      }
    }
    for (const auto &[waveId, laneSet] : mergedLanes) {
      for (LaneId laneId : laneSet) {
        membershipRegistry.setLaneStatus(waveId, laneId, targetBlockId,
                                         LaneBlockStatus::Participating);
        // targetBlock.addParticipatingLane(waveId, laneId);
      }
    }
    targetBlock.setIsConverged(totalMergedLanes == threadgroupSize);
  }

  // Reassign all lanes to the target block
  for (const auto &[waveId, laneSet] : mergedLanes) {
    for (LaneId laneId : laneSet) {
      // Update the BlockMembershipRegistry (single source of truth)
      membershipRegistry.setLaneStatus(waveId, laneId, targetBlockId,
                                       LaneBlockStatus::Participating);
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
  const auto *block = getBlock(blockId);
  if (block) {
    // Update BlockMembershipRegistry: lane arrived and is no longer unknown
    // First check current status to handle the transition properly
    LaneBlockStatus currentStatus =
        membershipRegistry.getLaneStatus(waveId, laneId, blockId);

    // If lane was unknown, it's now participating (arrived)
    if (currentStatus == LaneBlockStatus::Unknown) {
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::Participating);
      INTERPRETER_DEBUG_LOG("DEBUG: markLaneArrived - Lane " << laneId
                << " transitioned from Unknown to Participating in block "
                << blockId);
    } else {
      // Lane was already participating, just ensure it stays participating
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::Participating);
      INTERPRETER_DEBUG_LOG("DEBUG: markLaneArrived - Lane " << laneId
                << " confirmed as Participating in block " << blockId);
    }

    // Validate with registry
    bool registryResolved =
        membershipRegistry.isWaveAllUnknownResolved(waveId, blockId);
    bool oldResolved =
        membershipRegistry.getUnknownLanes(waveId, blockId).empty();

    if (registryResolved != oldResolved) {
      INTERPRETER_DEBUG_LOG("INFO: markLaneArrived - Block " << blockId << " wave "
                << waveId
                << " resolution difference - registry: " << registryResolved
                << ", old: " << oldResolved << " (tracked by registry)"
      );
    }

    // Resolution status tracked by registry - no need for old system metadata
  }
}

void ThreadgroupContext::markLaneWaitingForWave(WaveId waveId, LaneId laneId,
                                                uint32_t blockId) {
  INTERPRETER_DEBUG_LOG("DEBUG: markLaneWaitingForWave - Lane " << laneId << " wave "
            << waveId << " in block " << blockId);

  const auto *block = getBlock(blockId);
  if (block) {
    // it->second.addWaitingLane(waveId, laneId);
    // Change lane state to waiting
    if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
      waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;
      // Also update the BlockMembershipRegistry
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::WaitingForWave);
      INTERPRETER_DEBUG_LOG("DEBUG: markLaneWaitingForWave - Successfully set lane "
                << laneId << " to WaitingForWave in block " << blockId
      );
    }
  } else {
    INTERPRETER_DEBUG_LOG("DEBUG: markLaneWaitingForWave - Block " << blockId
              << " not found!");
  }
  // debugging purpose
  membershipRegistry.getCurrentBlock(waveId, laneId);
}

bool ThreadgroupContext::canExecuteWaveOperation(WaveId waveId,
                                                 LaneId laneId) const {
  uint32_t blockId = membershipRegistry.getCurrentBlock(waveId, laneId);
  if (blockId == 0) {
    return false; // Lane not in any block
  }
  const auto *block = getBlock(blockId);
  if (!block) {
    return false; // Block not found
  }

  // Can execute if all unknown lanes are resolved (we know the complete
  // participant set)
  auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
  return unknownLanes.empty();
}

std::vector<LaneId>
ThreadgroupContext::getWaveOperationParticipants(WaveId waveId,
                                                 LaneId laneId) const {
  uint32_t blockId = membershipRegistry.getCurrentBlock(waveId, laneId);
  if (blockId == 0) {
    return {}; // Lane not in any block
  }
  const auto *block = getBlock(blockId);
  if (!block) {
    return {}; // Block not found
  }

  // Return all lanes from the same wave that are CURRENTLY participating in
  // this block Use participating lanes instead of arrived lanes to avoid stale
  // membership
  std::vector<LaneId> participants;
  auto participatingLanes =
      membershipRegistry.getParticipatingLanes(waveId, blockId);
  for (LaneId participantId : participatingLanes) {
    if (participantId < waves[waveId]->lanes.size() &&
        waves[waveId]->lanes[participantId]->isActive &&
        !waves[waveId]->lanes[participantId]->hasReturned) {
      participants.push_back(participantId);
    }
  }

  return participants;
}

// TODO: check the functional correctness
// Block deduplication methods
uint32_t ThreadgroupContext::findOrCreateBlockForPath(
    const BlockIdentity &identity,
    const std::map<WaveId, std::set<LaneId>> &unknownLanes) {

  INTERPRETER_DEBUG_LOG("DEBUG: findOrCreateBlockForPath called with "
            << unknownLanes.size() << " waves of unknown lanes");
  for (const auto &[waveId, laneSet] : unknownLanes) {
    INTERPRETER_DEBUG_LOG("  Wave " << waveId << ": {");
    for (LaneId laneId : laneSet) {
      INTERPRETER_DEBUG_LOG(laneId << " ");
    }
    INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)");
  }

  // Check if block with this identity already exists
  auto it = identityToBlockId.find(identity);
  if (it != identityToBlockId.end()) {
    // Block exists - don't modify its unknown lanes!
    // The existing block already has the correct unknown lanes from when it was
    // first created. Those lanes will be properly removed by
    // removeThreadFromNestedBlocks when appropriate.
    uint32_t existingBlockId = it->second;
    INTERPRETER_DEBUG_LOG("DEBUG: Found existing block " << existingBlockId
              << " - not modifying unknown lanes");
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
  for (const auto &[waveId, laneSet] : unknownLanes) {
    for (LaneId laneId : laneSet) {
      INTERPRETER_DEBUG_LOG("DEBUG: addUnknownLane - adding lane " << laneId
                << " to new block " << newBlockId);
      // Update both old system and registry for consistency
      // newBlock.addUnknownLane(waveId, laneId);
      membershipRegistry.setLaneStatus(waveId, laneId, newBlockId,
                                       LaneBlockStatus::Unknown);
    }
  }

  // Resolution status for all waves is tracked by registry - no metadata needed

  newBlock.setIsConverged(false); // Will be updated as lanes arrive

  // Store the new block
  executionBlocks[newBlockId] = newBlock;
  identityToBlockId[identity] = newBlockId;
  
  // Register block type with the membership registry
  membershipRegistry.registerBlock(newBlockId, identity.blockType);

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
    const std::vector<const void *> &executionPath) const {
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
  instrIdentity.instructionType = "WaveActiveOp"; // Generic type for now
  instrIdentity.sourceExpression = instruction;

  // Use the wave-specific approach for instruction synchronization
  bool canExecuteInBlock =
      canExecuteWaveInstructionInBlock(currentBlockId, waveId, instrIdentity);

  // Debug unknown lanes state
  const auto *block = getBlock(currentBlockId);
  if (block) {
    auto unknownLanes =
        membershipRegistry.getUnknownLanes(waveId, currentBlockId);
    INTERPRETER_DEBUG_LOG("DEBUG: Block " << currentBlockId << " wave " << waveId
              << " unknown lanes: {");
    for (LaneId uid : unknownLanes) {
      INTERPRETER_DEBUG_LOG(uid << " ");
    }
    INTERPRETER_DEBUG_LOG("}");
  }

  // Check using compound key and state machine for the new system
  std::pair<const void *, uint32_t> instructionKey = {instruction,
                                                      currentBlockId};

  // Update sync point phase if it exists
  auto syncPointIt = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (syncPointIt != waves[waveId]->activeSyncPoints.end()) {
    syncPointIt->second.updatePhase(*this, waveId);
  }

  bool allParticipantsKnown =
      areAllParticipantsKnownForWaveInstruction(waveId, instructionKey);
  bool allParticipantsArrived =
      haveAllParticipantsArrivedAtWaveInstruction(waveId, instructionKey);
  bool canExecuteGlobal = allParticipantsKnown && allParticipantsArrived;

  std::string phaseStr = "no_sync_point";
  if (syncPointIt != waves[waveId]->activeSyncPoints.end()) {
    int phase = (int)syncPointIt->second.getPhase();
    phaseStr = "phase_" + std::to_string(phase);
  }

  INTERPRETER_DEBUG_LOG("DEBUG: canExecuteWaveInstruction for lane " << laneId
            << " in block " << currentBlockId
            << ": canExecuteInBlock=" << canExecuteInBlock
            << ", allParticipantsKnown=" << allParticipantsKnown
            << ", allParticipantsArrived=" << allParticipantsArrived
            << ", canExecuteGlobal=" << canExecuteGlobal
            << ", syncPointPhase=" << phaseStr);

  // Both approaches should agree for proper synchronization
  return canExecuteInBlock && canExecuteGlobal;
}

void ThreadgroupContext::markLaneWaitingAtWaveInstruction(
    WaveId waveId, LaneId laneId, const void *instruction,
    const std::string &instructionType, int waveOpType) {
  // Create instruction identity
  InstructionIdentity instrIdentity =
      createInstructionIdentity(instruction, instructionType, instruction);

  // Get current block for this lane
  uint32_t currentBlockId = getCurrentBlock(waveId, laneId);
  if (currentBlockId == 0) {
    INTERPRETER_DEBUG_LOG("ERROR: Lane " << laneId << " in wave " << waveId
              << " not assigned to any block when arriving at instruction "
              << instructionType);
    // Debug: Check registry state
    INTERPRETER_DEBUG_LOG("DEBUG: Checking all blocks for lane " << laneId << "...");
    for (const auto &[blockId, block] : executionBlocks) {
      auto status = membershipRegistry.getLaneStatus(waveId, laneId, blockId);
      if (status != LaneBlockStatus::Unknown &&
          status != LaneBlockStatus::Left) {
        INTERPRETER_DEBUG_LOG("  Found lane in block " << blockId << " with status "
                  << (int)status);
      }
    }
    throw std::runtime_error(
        "Lane not assigned to any block when arriving at instruction");
  }

  // Add instruction to block with this lane as participant
  std::map<WaveId, std::set<LaneId>> participants = {{waveId, {laneId}}};
  addInstructionToBlock(currentBlockId, instrIdentity, participants);

  // Create or update sync point for this instruction
  createOrUpdateWaveSyncPoint(instruction, waveId, laneId, instructionType, waveOpType);

  // Create compound key using existing currentBlockId
  std::pair<const void *, uint32_t> instructionKey = {instruction,
                                                      currentBlockId};

  auto &syncPoint = waves[waveId]->activeSyncPoints[instructionKey];
  syncPoint.arrivedParticipants.insert(laneId);

  // Mark lane as waiting at this instruction with compound key
  waves[waveId]->laneWaitingAtInstruction[laneId] = instructionKey;
  waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;

  // Also update the BlockMembershipRegistry
  membershipRegistry.setLaneStatus(waveId, laneId, currentBlockId,
                                   LaneBlockStatus::WaitingForWave);

  // Completion status is now computed on-demand via methods
}

bool ThreadgroupContext::areAllParticipantsKnownForWaveInstruction(
    WaveId waveId,
    const std::pair<const void *, uint32_t> &instructionKey) const {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    // No sync point created yet - check if only one lane is in the block
    uint32_t blockId = instructionKey.second;
    const auto *block = getBlock(blockId);
    if (block) {
      auto arrivedLanes = membershipRegistry.getArrivedLanes(waveId, blockId);
      if (arrivedLanes.size() == 1) {
        return true; // Single lane can proceed immediately
      }
    }
    return false;
  }

  // Check if all participants are known by querying the sync point method
  bool syncPointKnown = it->second.isAllParticipantsKnown(*this, waveId);

  // Cross-validate with BlockMembershipRegistry for consistency
  uint32_t blockId = instructionKey.second;
  bool registryKnown =
      membershipRegistry.isWaveAllUnknownResolved(waveId, blockId);

  if (syncPointKnown != registryKnown) {
    INTERPRETER_DEBUG_LOG("WARNING: Consistency mismatch - syncPoint says "
              << syncPointKnown << " but registry says " << registryKnown
              << " for wave " << waveId << " block " << blockId);
  }

  // Continue validation but keep sync point as authority for now
  bool finalResult = syncPointKnown; // Keep stable behavior

  // Log any discrepancies for monitoring
  if (syncPointKnown != registryKnown) {
    INTERPRETER_DEBUG_LOG("INFO: Participants known - sync point: " << syncPointKnown
              << ", registry: " << registryKnown << " (using sync point)");
  }

  if (finalResult) {
    INTERPRETER_DEBUG_LOG("DEBUG: areAllParticipantsKnownForWaveInstruction - All "
                 "participants known for sync point");
  }

  return finalResult;
}

bool ThreadgroupContext::haveAllParticipantsArrivedAtWaveInstruction(
    WaveId waveId,
    const std::pair<const void *, uint32_t> &instructionKey) const {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    return false; // No sync point created yet
  }

  return it->second.isAllParticipantsArrived();
}

// getWaveInstructionParticipants removed - functionality replaced by compound
// key system

void ThreadgroupContext::createOrUpdateWaveSyncPoint(
    const void *instruction, WaveId waveId, LaneId laneId,
    const std::string &instructionType, int waveOpType) {
  // Create compound key with current block ID
  uint32_t blockId = getCurrentBlock(waveId, laneId);
  std::pair<const void *, uint32_t> instructionKey = {instruction, blockId};

  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);

  if (it == waves[waveId]->activeSyncPoints.end()) {
    // Create new sync point
    WaveOperationSyncPoint syncPoint;
    syncPoint.instruction = instruction;
    syncPoint.instructionType = instructionType;
    syncPoint.waveOpType = waveOpType;

    // Use blockId from compound key
    syncPoint.blockId = blockId;

    // Get expected participants from the block
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }

    // Participants knowledge will be computed on-demand via
    // isAllParticipantsKnown()

    waves[waveId]->activeSyncPoints[instructionKey] = syncPoint;
  } else {
    // Update existing sync point
    auto &syncPoint = it->second;

    // Participants knowledge will be computed on-demand via
    // isAllParticipantsKnown()

    // Update expected participants if needed
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }
  }
}

void ThreadgroupContext::releaseWaveSyncPoint(
    WaveId waveId, const std::pair<const void *, uint32_t> &instructionKey) {
  auto it = waves[waveId]->activeSyncPoints.find(instructionKey);
  if (it != waves[waveId]->activeSyncPoints.end()) {
    const auto &syncPoint = it->second;

    // Only release sync points that are ready for cleanup
    if (!syncPoint.shouldCleanup()) {
      INTERPRETER_DEBUG_LOG("Sync point not ready for cleanup (phase: "
                            << (int)syncPoint.getPhase() << ")\n");
      return;
    }

    INTERPRETER_DEBUG_LOG("Releasing sync point: "
                          << syncPoint.arrivedParticipants.size()
                          << " participants (phase: "
                          << (int)syncPoint.getPhase() << ")\n");

    // Release all waiting lanes
    for (LaneId laneId : syncPoint.arrivedParticipants) {
      if (laneId < waves[waveId]->lanes.size()) {
        INTERPRETER_DEBUG_LOG("  Waking up lane "
                              << laneId << " (was "
                              << (waves[waveId]->lanes[laneId]->state ==
                                          ThreadState::WaitingForWave
                                      ? "WaitingForWave"
                                      : "other")
                              << ")\n");
        waves[waveId]->lanes[laneId]->state = ThreadState::Ready;
        waves[waveId]->lanes[laneId]->isResumingFromWaveOp =
            true; // Set resuming flag

        // CRITICAL FIX: Update registry status from WaitingForWave back to
        // Participating Use lane's current block, not stale block from
        // instruction key
        uint32_t currentBlockId = getCurrentBlock(waveId, laneId);
        membershipRegistry.onLaneFinishWaveOp(waveId, laneId, currentBlockId);
      }
      waves[waveId]->laneWaitingAtInstruction.erase(laneId);
    }

    // Remove the sync point
    waves[waveId]->activeSyncPoints.erase(it);
    INTERPRETER_DEBUG_LOG("  Sync point removed from active list\n");
  } else {
    INTERPRETER_DEBUG_LOG("WARNING: Sync point not found for instruction "
                          << instructionKey.first << " block "
                          << instructionKey.second << "\n");
  }
}

// Merge stack management methods
void ThreadgroupContext::pushMergePoint(
    WaveId waveId, LaneId laneId, const void *sourceStmt,
    uint32_t parentBlockId, const std::set<uint32_t> &divergentBlocks) {
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
    WaveId waveId, LaneId laneId,
    const std::vector<MergeStackEntry> &mergeStack) {
  waves[waveId]->laneMergeStacks[laneId] = mergeStack;
}

// Instruction identity management methods
void ThreadgroupContext::addInstructionToBlock(
    uint32_t blockId, const InstructionIdentity &instruction,
    const std::map<WaveId, std::set<LaneId>> &participants) {
  auto *block = getBlock(blockId);
  if (!block) {
    return; // Block doesn't exist
  }

  // Add instruction to the ordered list if not already present
  bool found = false;
  for (const auto &existingInstr : block->getInstructionList()) {
    if (existingInstr == instruction) {
      found = true;
      break;
    }
  }

  if (!found) {
    block->addInstruction(instruction);
  }

  // Merge participants for this instruction (don't replace, add to existing)
  for (const auto &[waveId, newLanes] : participants) {
    for (LaneId laneId : newLanes) {
      block->addInstructionParticipant(instruction, waveId, laneId);
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
  const auto *block = getBlock(blockId);
  if (block) {
    return block->getInstructionList();
  }
  return {};
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getInstructionParticipantsInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  const auto *block = getBlock(blockId);
  if (block) {
    const auto &instructionParticipants = block->getInstructionParticipants();
    auto instrIt = instructionParticipants.find(instruction);
    if (instrIt != instructionParticipants.end()) {
      return instrIt->second;
    }
  }
  return {};
}

bool ThreadgroupContext::canExecuteWaveInstructionInBlock(
    uint32_t blockId, WaveId waveId,
    const InstructionIdentity &instruction) const {
  const auto *block = getBlock(blockId);
  if (!block) {
    return false; // Block doesn't exist
  }
  // For wave operations, only check if unknown lanes for this specific wave are
  // resolved
  auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
  if (!unknownLanes.empty()) {
    return false; // Still have unknown lanes for this wave
  }

  return true; // Wave operation can proceed when all lanes from this wave are
               // known
}

bool ThreadgroupContext::canExecuteBarrierInstructionInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  const auto *block = getBlock(blockId);
  if (!block) {
    return false; // Block doesn't exist
  }

  // For barriers, check if all unknown lanes across all waves are resolved
  // (using registry)
  for (uint32_t waveId = 0; waveId < waveCount; ++waveId) {
    auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
    if (!unknownLanes.empty()) {
      return false; // Still have unknown lanes in this wave
    }
  }

  // Get expected participants (all active lanes in the block) - reconstruct
  // from registry
  std::map<WaveId, std::set<LaneId>> expectedParticipants;
  for (uint32_t waveId = 0; waveId < waveCount; ++waveId) {
    auto participatingLanes =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    if (!participatingLanes.empty()) {
      expectedParticipants[waveId] = participatingLanes;
    }
  }

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

// Proactive block creation methods
std::tuple<uint32_t, uint32_t, uint32_t> ThreadgroupContext::createIfBlocks(
    const void *ifStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack, bool hasElse,
    const std::vector<const void *> &executionPath) {
  INTERPRETER_DEBUG_LOG("DEBUG: createIfBlocks - ifStmt=" << ifStmt
            << ", parentBlockId=" << parentBlockId << ", hasElse=" << hasElse);
  INTERPRETER_DEBUG_LOG("DEBUG: createIfBlocks - mergeStack size=" << mergeStack.size());
  for (size_t i = 0; i < mergeStack.size(); i++) {
    INTERPRETER_DEBUG_LOG("  MergeStack[" << i
              << "]: sourceStatement=" << mergeStack[i].sourceStatement);
  }
  INTERPRETER_DEBUG_LOG("DEBUG: createIfBlocks - executionPath size="
            << executionPath.size());
  for (size_t i = 0; i < executionPath.size(); i++) {
    INTERPRETER_DEBUG_LOG("  ExecutionPath[" << i << "]=" << executionPath[i]);
  }

  // Get all lanes that could potentially take either path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Always create the then block
  BlockIdentity thenIdentity =
      createBlockIdentity(ifStmt, BlockType::BRANCH_THEN, parentBlockId,
                          mergeStack, true, executionPath);
  uint32_t thenBlockId =
      findOrCreateBlockForPath(thenIdentity, allPotentialLanes);

  uint32_t elseBlockId = 0; // 0 means no else block

  // Only create else block if it exists in the code
  if (hasElse) {
    BlockIdentity elseIdentity =
        createBlockIdentity(ifStmt, BlockType::BRANCH_ELSE, parentBlockId,
                            mergeStack, false, executionPath);
    elseBlockId = findOrCreateBlockForPath(elseIdentity, allPotentialLanes);
  }

  // Always create merge block for reconvergence
  BlockIdentity mergeIdentity = createBlockIdentity(
      ifStmt, BlockType::MERGE, parentBlockId, mergeStack, true, executionPath);
  uint32_t mergeBlockId =
      findOrCreateBlockForPath(mergeIdentity, allPotentialLanes);

  INTERPRETER_DEBUG_LOG("DEBUG: createIfBlocks - Created blocks: thenBlockId="
            << thenBlockId << ", elseBlockId=" << elseBlockId
            << ", mergeBlockId=" << mergeBlockId);

  return {thenBlockId, elseBlockId, mergeBlockId};
}

void ThreadgroupContext::moveThreadFromUnknownToParticipating(uint32_t blockId,
                                                              WaveId waveId,
                                                              LaneId laneId) {
  auto *block = getBlock(blockId);
  if (!block)
    return;
  // Update lane assignment
  INTERPRETER_DEBUG_LOG("DEBUG: moveThreadFromUnknownToParticipating - moving lane "
            << laneId << " to block " << blockId);
  assignLaneToBlock(waveId, laneId, blockId);

  // Verify the assignment worked
  uint32_t newBlock = getCurrentBlock(waveId, laneId);
  INTERPRETER_DEBUG_LOG("DEBUG: moveThreadFromUnknownToParticipating - lane " << laneId
            << " is now in block " << newBlock);
}

void ThreadgroupContext::removeThreadFromUnknown(uint32_t blockId,
                                                 WaveId waveId, LaneId laneId) {
  auto *block = getBlock(blockId);
  if (!block)
    return;

  // Remove lane from unknown (it chose a different path)
  INTERPRETER_DEBUG_LOG("DEBUG: removeThreadFromUnknown - removing lane " << laneId
            << " from block " << blockId);

  // Show unknown lanes before removal
  auto unknownBefore = membershipRegistry.getUnknownLanes(waveId, blockId);
  INTERPRETER_DEBUG_LOG("DEBUG: Block " << blockId << " unknown lanes before removal: {");
  for (auto it = unknownBefore.begin(); it != unknownBefore.end(); ++it) {
    if (it != unknownBefore.begin())
      INTERPRETER_DEBUG_LOG(" ");
    INTERPRETER_DEBUG_LOG(*it);
  }
  INTERPRETER_DEBUG_LOG("}");

  // Update both old system and registry for consistency
  // block.removeUnknownLane(waveId, laneId);
  membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                   LaneBlockStatus::Left);

  // Show unknown lanes after removal
  auto unknownAfter = membershipRegistry.getUnknownLanes(waveId, blockId);
  INTERPRETER_DEBUG_LOG("DEBUG: Block " << blockId << " unknown lanes after removal: {");
  for (auto it = unknownAfter.begin(); it != unknownAfter.end(); ++it) {
    if (it != unknownAfter.begin())
      INTERPRETER_DEBUG_LOG(" ");
    INTERPRETER_DEBUG_LOG(*it);
  }
  INTERPRETER_DEBUG_LOG("}");

  // Update resolution status
  auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
  bool allResolved = unknownLanes.empty();
  INTERPRETER_DEBUG_LOG("DEBUG: Block " << blockId
            << " areAllUnknownLanesResolvedForWave(" << waveId
            << ") = " << allResolved << " (tracked by registry)");
  // Resolution status tracked by registry - no need for old system metadata

  // CRITICAL: Check if removing this unknown lane allows any waiting wave
  // operations to proceed This handles the case where lanes 0,1 are waiting in
  // block 2 for a wave op, and lanes 2,3 choosing the else branch resolves all
  // unknowns
  if (unknownLanes.empty()) {
    INTERPRETER_DEBUG_LOG("DEBUG: Block " << blockId
              << " now has all unknowns resolved for wave " << waveId
              << " - checking for ready wave operations");

    // Check all waiting lanes in this block to see if their wave operations can
    // now proceed
    auto waitingLanes = membershipRegistry.getWaitingLanes(waveId, blockId);
    for (LaneId waitingLaneId : waitingLanes) {
      // Check if this lane is waiting for a wave operation in this block
      if (waves[waveId]->laneWaitingAtInstruction.count(waitingLaneId)) {
        auto &instructionKey =
            waves[waveId]->laneWaitingAtInstruction[waitingLaneId];
        // Only process if the instruction is in this block
        if (instructionKey.second == blockId) {
          INTERPRETER_DEBUG_LOG("DEBUG: Checking if waiting lane " << waitingLaneId
                    << " can now execute wave operation in block " << blockId
          );

          // Re-evaluate the sync point for this instruction
          auto &syncPoint = waves[waveId]->activeSyncPoints[instructionKey];
          // Sync point state is now computed on-demand

          if (syncPoint.isAllParticipantsArrived() &&
              syncPoint.isAllParticipantsKnown(*this, waveId)) {
            INTERPRETER_DEBUG_LOG("DEBUG: Wave operation in block " << blockId
                      << " is now ready to execute - all participants known "
                         "and arrived!"
            );
          }
        }
      }
    }
  }
}

// Helper method to completely remove a lane from all sets of a specific block
void ThreadgroupContext::removeThreadFromAllSets(uint32_t blockId,
                                                 WaveId waveId, LaneId laneId) {

  const auto *block = getBlock(blockId);
  if (block) {
    INTERPRETER_DEBUG_LOG("DEBUG: removeThreadFromAllSets - removing lane " << laneId
              << " from all sets of block " << blockId);

    // Also update the BlockMembershipRegistry
    auto partLanesBefore =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    auto waitingLanesBefore =
        membershipRegistry.getWaitingLanes(waveId, blockId);
    // size_t laneCountBefore = partLanesBefore.size() + waitingLanesBefore.size();

    // Show which lanes are in the block before removal
    // INTERPRETER_DEBUG_LOG("DEBUG: removeThreadFromAllSets - block " << blockId << " had "
    //           << laneCountBefore << " participating lanes before removal");
    // if (!partLanesBefore.empty()) {
    //   std::cout << " (participating: ";
    //   for (auto lid : partLanesBefore) {
    //     std::cout << lid << " ";
    //   }
    //   std::cout << ")";
    // }
    // if (!waitingLanesBefore.empty()) {
    //   std::cout << " (waiting: ";
    //   for (auto lid : waitingLanesBefore) {
    //     std::cout << lid << " ";
    //   }
    //   std::cout << ")";
    // }
    // std::cout << std::endl;

    membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                     LaneBlockStatus::Left);

    auto partLanesAfter =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    auto waitingLanesAfter =
        membershipRegistry.getWaitingLanes(waveId, blockId);
    // size_t laneCountAfter = partLanesAfter.size() + waitingLanesAfter.size();

    // Show which lanes are in the block after removal
  //   std::cout << "DEBUG: removeThreadFromAllSets - block " << blockId << " has "
  //             << laneCountAfter << " participating lanes after removal";
  //   if (!partLanesAfter.empty()) {
  //     std::cout << " (participating: ";
  //     for (auto lid : partLanesAfter) {
  //       std::cout << lid << " ";
  //     }
  //     std::cout << ")";
  //   }
  //   if (!waitingLanesAfter.empty()) {
  //     std::cout << " (waiting: ";
  //     for (auto lid : waitingLanesAfter) {
  //       std::cout << lid << " ";
  //     }
  //     std::cout << ")";
  //   }
  //   std::cout << std::endl;
  }
}

// TODO: verify the functional correctness of this method
void ThreadgroupContext::removeThreadFromNestedBlocks(uint32_t parentBlockId,
                                                      WaveId waveId,
                                                      LaneId laneId) {
  // Find all blocks that are nested within the parent block and remove the lane
  for (auto &[blockId, block] : executionBlocks) {
    if (block.getParentBlockId() == parentBlockId) {
      // Skip LOOP_EXIT/MERGE blocks - those are where lanes go when they exit
      // the loop! if (block.getBlockType() == BlockType::LOOP_EXIT ||
      // block.getBlockType() == BlockType::MERGE) {
      if (block.getBlockType() == BlockType::LOOP_EXIT) {
        INTERPRETER_DEBUG_LOG("DEBUG: removeThreadFromNestedBlocks - skipping "
                  << blockId << " (LOOP_EXIT block where lanes should go)"
        );
        continue;
      }

      // This is a direct child of the parent block - remove from all sets
      INTERPRETER_DEBUG_LOG("DEBUG: removeThreadFromNestedBlocks - removing lane "
                << laneId << " from block " << blockId << " (child of "
                << parentBlockId << ")");
      removeThreadFromAllSets(blockId, waveId, laneId);

      // Recursively remove from nested blocks of this child
      removeThreadFromNestedBlocks(blockId, waveId, laneId);
    }
  }
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getCurrentBlockParticipants(uint32_t blockId) const {
  const auto *block = getBlock(blockId);
  if (block) {
    // Return union of participating, arrived, waiting, and unknown lanes
    std::map<WaveId, std::set<LaneId>> allLanes;

    for (const auto &[waveId, lanes] : block->getParticipatingLanes(*this)) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block->getArrivedLanes(*this)) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block->getWaitingLanes(*this)) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block->getUnknownLanes(*this)) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }

    return allLanes;
  }
  return {};
}

std::tuple<uint32_t, uint32_t> ThreadgroupContext::createLoopBlocks(
    const void *loopStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack,
    const std::vector<const void *> &executionPath) {
  // Get all lanes that could potentially enter the loop
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Create loop header block (where condition is checked) - parent is the block
  // before loop
  BlockIdentity headerIdentity = createBlockIdentity(
      loopStmt, BlockType::LOOP_HEADER, parentBlockId, mergeStack);
  uint32_t headerBlockId =
      findOrCreateBlockForPath(headerIdentity, allPotentialLanes);

  // Note: Body blocks are no longer pre-created here since ForStmt, WhileStmt,
  // and DoWhileStmt all create unique iteration body blocks during execution

  // Create loop exit/merge block - parent should be header block (where threads
  // reconverge after loop)
  BlockIdentity mergeIdentity = createBlockIdentity(
      loopStmt, BlockType::LOOP_EXIT, headerBlockId, mergeStack);
  uint32_t mergeBlockId =
      findOrCreateBlockForPath(mergeIdentity, allPotentialLanes);

  return {headerBlockId, mergeBlockId};
}

std::vector<uint32_t> ThreadgroupContext::createSwitchCaseBlocks(
    const void *switchStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack,
    const std::vector<int> &caseValues, bool hasDefault,
    const std::vector<const void *> &executionPath) {
  // Get all lanes that could potentially take any case path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Create switch header block for condition evaluation
  BlockIdentity headerIdentity =
      createBlockIdentity(switchStmt, BlockType::SWITCH_HEADER, parentBlockId,
                          mergeStack, true, executionPath);
  uint32_t headerBlockId =
      findOrCreateBlockForPath(headerIdentity, allPotentialLanes);

  std::vector<uint32_t> allBlockIds;
  allBlockIds.push_back(headerBlockId); // First element is header block

  std::vector<uint32_t> caseBlockIds;

  // Create a block for each case value (parent is header block)
  for (size_t i = 0; i < caseValues.size(); ++i) {
    // Create unique identity by including case value in the statement pointer
    const void *uniqueCasePtr = reinterpret_cast<const void *>(
        reinterpret_cast<uintptr_t>(switchStmt) + caseValues[i] + 1);
    BlockIdentity caseIdentity =
        createBlockIdentity(uniqueCasePtr, BlockType::SWITCH_CASE,
                            headerBlockId, mergeStack, true, executionPath);
    uint32_t caseBlockId =
        findOrCreateBlockForPath(caseIdentity, allPotentialLanes);
    caseBlockIds.push_back(caseBlockId);
    allBlockIds.push_back(caseBlockId);
  }

  // Create default block if it exists (parent is header block)
  if (hasDefault) {
    BlockIdentity defaultIdentity =
        createBlockIdentity(switchStmt, BlockType::SWITCH_DEFAULT,
                            headerBlockId, mergeStack, true, executionPath);
    uint32_t defaultBlockId =
        findOrCreateBlockForPath(defaultIdentity, allPotentialLanes);
    caseBlockIds.push_back(defaultBlockId);
    allBlockIds.push_back(defaultBlockId);
  }

  // Create merge block for switch reconvergence (parent is header block)
  BlockIdentity mergeIdentity =
      createBlockIdentity(switchStmt, BlockType::SWITCH_MERGE, headerBlockId,
                          mergeStack, true, executionPath);
  uint32_t mergeBlockId =
      findOrCreateBlockForPath(mergeIdentity, allPotentialLanes);

  // Return: [headerBlockId, caseBlock1, caseBlock2, ..., mergeBlockId]
  allBlockIds.push_back(mergeBlockId);
  return allBlockIds;
}

// Debug and visualization methods implementation
void ThreadgroupContext::printDynamicExecutionGraph(bool verbose) const {
  std::cout <<
      "\n=== Dynamic Execution Graph (MiniHLSL Interpreter) ===\n";
  std::cout << "Threadgroup Size: " << threadgroupSize << "\n";
  std::cout << "Wave Size: " << waveSize << "\n";
  std::cout << "Wave Count: " << waveCount << "\n";
  std::cout << "Total Dynamic Blocks: " << executionBlocks.size()
                                                 << "\n";
  std::cout << "Next Block ID: " << nextBlockId << "\n\n";

  // Print each dynamic execution block
  for (const auto &[blockId, block] : executionBlocks) {
    printBlockDetails(blockId, verbose);
    std::cout << "\n";
  }

  if (verbose) {
    std::cout << "=== Wave States ===\n";
    for (uint32_t waveId = 0; waveId < waveCount; ++waveId) {
      printWaveState(waveId, verbose);
      std::cout << "\n";
    }
  }

  std::cout << "=== End Dynamic Execution Graph ===\n\n";
}

void ThreadgroupContext::printBlockDetails(uint32_t blockId,
                                           bool verbose) const {
  const auto *block = getBlock(blockId);
  if (!block) {
    std::cout << "Block " << blockId << ": NOT FOUND\n";
    return;
  }
  std::cout << "Dynamic Block " << blockId << ":\n";

  // Basic block info
  std::cout << "  Block ID: " << block->getBlockId() << "\n";

  // Show block type from identity
  const auto &identity = block->getIdentity();
  const char *blockTypeName = "UNKNOWN";
  switch (identity.blockType) {
  case BlockType::REGULAR:
    blockTypeName = "REGULAR";
    break;
  case BlockType::BRANCH_THEN:
    blockTypeName = "BRANCH_THEN";
    break;
  case BlockType::BRANCH_ELSE:
    blockTypeName = "BRANCH_ELSE";
    break;
  case BlockType::MERGE:
    blockTypeName = "MERGE";
    break;
  case BlockType::LOOP_HEADER:
    blockTypeName = "LOOP_HEADER";
    break;
  // case BlockType::LOOP_BODY:
  //   blockTypeName = "LOOP_BODY";
  //   break;
  case BlockType::LOOP_EXIT:
    blockTypeName = "LOOP_EXIT";
    break;
  case BlockType::SWITCH_HEADER:
    blockTypeName = "SWITCH_HEADER";
    break;
  case BlockType::SWITCH_CASE:
    blockTypeName = "SWITCH_CASE";
    break;
  case BlockType::SWITCH_DEFAULT:
    blockTypeName = "SWITCH_DEFAULT";
    break;
  case BlockType::SWITCH_MERGE:
    blockTypeName = "SWITCH_MERGE";
    break;
  }
  std::cout <<"  Block Type: " << blockTypeName << "\n";
  std::cout <<"  Parent Block: " << block->getParentBlockId()
                                           << "\n";
  std::cout <<"  Program Point: " << block->getProgramPoint()
                                            << "\n";
  std::cout <<
      "  Is Converged: " << (block->getIsConverged() ? "Yes" : "No") << "\n";
  std::cout <<"  Nesting Level: " << block->getNestingLevel()
                                            << "\n";

  // Source statement info
  if (block->getSourceStatement()) {
    std::cout <<
        "  Source Statement: "
        << static_cast<const void *>(block->getSourceStatement()) << "\n";
  }

  // Participating lanes by wave
  size_t totalLanes = 0;
  std::cout <<"  Participating Lanes by Wave:\n";
  for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
    auto laneSet = membershipRegistry.getParticipatingLanes(waveId, blockId);
    std::cout <<"    Wave " << waveId << ": {";
    bool first = true;
    for (LaneId laneId : laneSet) {
      if (!first)
        std::cout <<", ";
      std::cout <<laneId;
      first = false;
    }
    std::cout <<"} (" << laneSet.size() << " lanes)\n";
    totalLanes += laneSet.size();
  }
  std::cout <<"  Total Participating Lanes: " << totalLanes << "\n";

  if (verbose) {
    // Unknown lanes
    bool hasUnknownLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
      if (!unknownLanes.empty()) {
        if (!hasUnknownLanes) {
          std::cout <<"  Unknown Lanes by Wave:\n";
          hasUnknownLanes = true;
        }
        auto laneSet = unknownLanes;
        std::cout <<"    Wave " << waveId << ": {";
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            std::cout <<", ";
          std::cout <<laneId;
          first = false;
        }
        std::cout <<"} (" << laneSet.size() << " lanes)\n";
      }
    }

    // Arrived lanes
    bool hasArrivedLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto arrivedLanes = membershipRegistry.getArrivedLanes(waveId, blockId);
      if (!arrivedLanes.empty()) {
        if (!hasArrivedLanes) {
          std::cout <<"  Arrived Lanes by Wave:\n";
          hasArrivedLanes = true;
        }
        auto laneSet = arrivedLanes;
        std::cout <<"    Wave " << waveId << ": {";
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            std::cout <<", ";
          std::cout <<laneId;
          first = false;
        }
        std::cout <<"} (" << laneSet.size() << " lanes)\n";
      }
    }

    // Waiting lanes
    bool hasWaitingLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto waitingLanes = membershipRegistry.getWaitingLanes(waveId, blockId);
      if (!waitingLanes.empty()) {
        if (!hasWaitingLanes) {
          std::cout <<"  Waiting Lanes by Wave:\n";
          hasWaitingLanes = true;
        }
        auto laneSet = waitingLanes;
        std::cout <<"    Wave " << waveId << ": {";
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            std::cout <<", ";
          std::cout <<laneId;
          first = false;
        }
        std::cout <<"} (" << laneSet.size() << " lanes)\n";
      }
    }

    // Instructions in this block
    const auto &instructions = block->getInstructionList();
    if (!instructions.empty()) {
      std::cout <<"  Instructions (" << instructions.size()
                                               << "):\n";
      for (size_t i = 0; i < instructions.size(); ++i) {
        const auto &instr = instructions[i];
        std::cout <<"    " << i << ": " << instr.instructionType
                                     << " (ptr: " << instr.instruction
                                     << ")\n";
      }
    }
  }
}

void ThreadgroupContext::printWaveState(WaveId waveId, bool verbose) const {
  if (waveId >= waves.size()) {
    std::cout <<"Wave " << waveId << ": NOT FOUND\n";
    return;
  }

  const auto &wave = waves[waveId];
  std::cout <<"Wave " << waveId << ":\n";
  std::cout <<"  Wave Size: " << wave->waveSize << "\n";
  std::cout <<"  Lane Count: " << wave->lanes.size() << "\n";
  std::cout <<"  Active Lanes: " << wave->countActiveLanes() << "\n";
  std::cout <<"  Currently Active Lanes: "
                        << wave->countCurrentlyActiveLanes() << "\n";

  if (verbose) {
    // Lane to block mapping
    std::cout <<"  Lane to Block Mapping (from registry):\n";
    for (LaneId laneId = 0; laneId < wave->lanes.size(); ++laneId) {
      uint32_t blockId = getCurrentBlock(waveId, laneId);
      if (blockId != 0) {
        std::cout <<"    Lane " << laneId << " -> Block " << blockId
                                          << "\n";
      }
    }

    // Active sync points
    if (!wave->activeSyncPoints.empty()) {
      std::cout <<"  Active Sync Points ("
                            << wave->activeSyncPoints.size() << "):\n";
      for (const auto &[instructionKey, syncPoint] : wave->activeSyncPoints) {
        std::cout <<"    Instruction "
                              << instructionKey.first << " block "
                              << instructionKey.second << " ("
                              << syncPoint.instructionType << "):\n";
        std::cout <<"      Expected: "
                              << syncPoint.expectedParticipants.size()
                              << " lanes\n";
        std::cout <<"      Arrived: "
                              << syncPoint.arrivedParticipants.size()
                              << " lanes\n";
        std::cout <<
            "      Ready to execute: "
            << (syncPoint.isReadyToExecute(*this, waveId) ? "Yes" : "No")
            << "\n";
      }
    }
  }
}

std::string ThreadgroupContext::getBlockSummary(uint32_t blockId) const {
  const auto *block = getBlock(blockId);
  if (!block) {
    return "Block " + std::to_string(blockId) + ": NOT FOUND";
  }
  size_t totalLanes = 0;
  for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
    auto participatingLanes =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    totalLanes += participatingLanes.size();
  }

  std::stringstream ss;
  ss << "Block " << blockId << " (Parent: " << block->getParentBlockId()
     << ", Lanes: " << totalLanes
     << ", Converged: " << (block->getIsConverged() ? "Y" : "N") << ")";
  return ss.str();
}

// Helper methods to safely get blocks and eliminate repetitive pattern
DynamicExecutionBlock *ThreadgroupContext::getBlock(uint32_t blockId) {
  auto it = executionBlocks.find(blockId);
  return (it != executionBlocks.end()) ? &it->second : nullptr;
}

const DynamicExecutionBlock *
ThreadgroupContext::getBlock(uint32_t blockId) const {
  auto it = executionBlocks.find(blockId);
  return (it != executionBlocks.end()) ? &it->second : nullptr;
}

void ThreadgroupContext::printFinalVariableValues() const {
  std::cout <<"\n=== Final Variable Values ===\n";

  for (size_t waveId = 0; waveId < waves.size(); ++waveId) {
    const auto &wave = waves[waveId];
    std::cout <<"Wave " << waveId << ":\n";

    for (size_t laneId = 0; laneId < wave->lanes.size(); ++laneId) {
      const auto &lane = wave->lanes[laneId];
      std::cout <<"  Lane " << laneId << ":\n";

      // Print all variables for this lane
      if (lane->variables.empty()) {
        std::cout <<"    (no variables)\n";
      } else {
        for (const auto &[varName, value] : lane->variables) {
          std::cout <<"    " << varName << " = " << value.toString()
                                       << "\n";
        }
      }

      // Print return value if present
      if (lane->hasReturned) {
        std::cout <<"    (returned: " << lane->returnValue.toString()
                                                << ")\n";
      }

      // Print thread state
      const char *stateStr = "Unknown";
      switch (lane->state) {
      case ThreadState::Ready:
        stateStr = "Ready";
        break;
      case ThreadState::WaitingAtBarrier:
        stateStr = "WaitingAtBarrier";
        break;
      case ThreadState::WaitingForWave:
        stateStr = "WaitingForWave";
        break;
      case ThreadState::WaitingForResume:
        stateStr = "WaitingForResume";
        break;
      case ThreadState::Completed:
        stateStr = "Completed";
        break;
      case ThreadState::Error:
        stateStr = "Error";
        break;
      }
      std::cout <<"    (state: " << stateStr << ")\n";

      // Print error message if in error state
      if (lane->state == ThreadState::Error && !lane->errorMessage.empty()) {
        std::cout <<"    (error: " << lane->errorMessage << ")\n";
      }
    }
  }

  std::cout <<"=== End Variable Values ===\n\n";
}

} // namespace interpreter
} // namespace minihlsl