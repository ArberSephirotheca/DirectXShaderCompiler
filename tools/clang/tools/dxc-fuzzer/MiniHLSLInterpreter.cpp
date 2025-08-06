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

static constexpr bool ENABLE_INTERPRETER_DEBUG =
    true; // Set to true to enable detailed execution tracing
static constexpr bool ENABLE_WAVE_DEBUG =
    true; // Set to true to enable wave operation tracing
static constexpr bool ENABLE_BLOCK_DEBUG =
    true; // Set to true to enable block lifecycle tracing

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

namespace minihlsl {
namespace interpreter {

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
  if (std::holds_alternative<int>(oldValue.data) && std::holds_alternative<int>(value.data)) {
    data_[index] = Value(std::get<int>(oldValue.data) & std::get<int>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicOr(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int>(oldValue.data) && std::holds_alternative<int>(value.data)) {
    data_[index] = Value(std::get<int>(oldValue.data) | std::get<int>(value.data));
  }
  return oldValue;
}

Value GlobalBuffer::atomicXor(uint32_t index, const Value &value) {
  if (index >= size_)
    return Value(0);

  Value oldValue = data_[index];
  if (std::holds_alternative<int>(oldValue.data) && std::holds_alternative<int>(value.data)) {
    data_[index] = Value(std::get<int>(oldValue.data) ^ std::get<int>(value.data));
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
      std::cout << "DEBUG: Initializing wave " << w << " with "
                << allLanes.size() << " lanes in initial block "
                << initialBlockId << std::endl;
      for (LaneId laneId : allLanes) {
        markLaneArrived(w, laneId, initialBlockId);
        // Verify lane was assigned
        uint32_t assignedBlock = membershipRegistry.getCurrentBlock(w, laneId);
        std::cout << "DEBUG: Lane " << laneId << " assigned to block "
                  << assignedBlock << std::endl;
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

uint32_t BlockMembershipRegistry::getCurrentBlock(uint32_t waveId,
                                                  LaneId laneId) const {
  // Find the block where this lane is currently participating or waiting
  // Prefer non-header blocks over header blocks for dual membership scenarios
  uint32_t headerBlock = 0;
  uint32_t nonHeaderBlock = 0;

  for (const auto &[key, status] : membership_) {
    if (std::get<0>(key) == waveId && std::get<1>(key) == laneId) {
      if (status == LaneBlockStatus::Participating ||
          status == LaneBlockStatus::WaitingForWave) {
        uint32_t blockId = std::get<2>(key);
        // TODO: We need block type info to distinguish header vs non-header
        // For now, use heuristic: higher block IDs are usually body blocks
        if (blockId > headerBlock) {
          nonHeaderBlock = blockId;
        } else if (headerBlock == 0) {
          headerBlock = blockId;
        }
      }
    }
  }

  // Return non-header block if found, otherwise header block
  return nonHeaderBlock != 0 ? nonHeaderBlock : headerBlock;
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
  std::cout << "DEBUG: onLaneFinishWaveOp - Lane " << laneId
            << " finishing wave op in block " << blockId << std::endl;

  // Check current status before update
  auto currentStatus = getLaneStatus(waveId, laneId, blockId);
  std::cout << "DEBUG: onLaneFinishWaveOp - Current status in block " << blockId
            << ": " << (int)currentStatus << std::endl;

  // Only update if lane is actually in this block as WaitingForWave
  if (currentStatus == LaneBlockStatus::WaitingForWave) {
    setLaneStatus(waveId, laneId, blockId, LaneBlockStatus::Participating);
    std::cout << "DEBUG: onLaneFinishWaveOp - Updated lane " << laneId
              << " to Participating in block " << blockId << std::endl;
  } else {
    std::cout << "WARNING: onLaneFinishWaveOp - Lane " << laneId
              << " not WaitingForWave in block " << blockId
              << ", skipping update" << std::endl;
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
  const auto* block = tg.getBlock(blockId);
  if (block) {
    // Check if all unknown lanes are resolved using registry
    auto unknownLanes = tg.membershipRegistry.getUnknownLanes(waveId, blockId);
    bool result = unknownLanes.empty();

    if (!result) {
      std::cout << "DEBUG: isAllParticipantsKnown - Block " << blockId
                << " wave " << waveId << " has " << unknownLanes.size()
                << " unknown lanes: ";
      for (auto laneId : unknownLanes) {
        std::cout << laneId << " ";
      }
      std::cout << " - These lanes need to be resolved to Participating or Left"
                << std::endl;
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
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName =
          varExpr->toString(); // Use toString() to get variable name
      Value &var = lane.variables[varName];
      var = Value(var.asInt() + 1);
      return var;
    }
    throw std::runtime_error("Pre-increment requires a variable");
  }
  case PostIncrement: {
    // i++: return old value, then increment
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName =
          varExpr->toString(); // Use toString() to get variable name
      Value &var = lane.variables[varName];
      Value oldValue = var;
      var = Value(var.asInt() + 1);
      return oldValue;
    }
    throw std::runtime_error("Post-increment requires a variable");
  }
  case PreDecrement: {
    // --i: decrement first, then return new value
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName =
          varExpr->toString(); // Use toString() to get variable name
      Value &var = lane.variables[varName];
      var = Value(var.asInt() - 1);
      return var;
    }
    throw std::runtime_error("Pre-decrement requires a variable");
  }
  case PostDecrement: {
    // i--: return old value, then decrement
    if (auto varExpr = dynamic_cast<const VariableExpr *>(expr_.get())) {
      std::string varName =
          varExpr->toString(); // Use toString() to get variable name
      Value &var = lane.variables[varName];
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

// Pure Result-based implementations for key expression types

Result<Value, ExecutionError> VariableExpr::evaluate_result(LaneContext &lane, WaveContext &,
                                                           ThreadgroupContext &) const {
  std::cout << "DEBUG: VariableExpr - Lane " << lane.laneId << " evaluating variable '" << name_ << "' (Result-based)" << std::endl;
  
  auto it = lane.variables.find(name_);
  if (it == lane.variables.end()) {
    std::cout << "DEBUG: VariableExpr - Variable '" << name_ << "' not found" << std::endl;
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
  
  std::cout << "DEBUG: VariableExpr - Variable '" << name_ << "' = " << it->second.toString() << std::endl;
  return Ok<Value, ExecutionError>(it->second);
}

Result<Value, ExecutionError> BinaryOpExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                           ThreadgroupContext &tg) const {
  std::cout << "DEBUG: BinaryOpExpr - Lane " << lane.laneId << " evaluating binary operation (Result-based)" << std::endl;
  
  // Evaluate left operand
  auto leftResult = left_->evaluate_result(lane, wave, tg);
  if (leftResult.is_err()) {
    return leftResult;
  }
  Value leftVal = leftResult.unwrap();
  
  // Evaluate right operand  
  auto rightResult = right_->evaluate_result(lane, wave, tg);
  if (rightResult.is_err()) {
    return rightResult;
  }
  Value rightVal = rightResult.unwrap();

  // Perform operation
  switch (op_) {
  case Add:
    return Ok<Value, ExecutionError>(leftVal + rightVal);
  case Sub:
    return Ok<Value, ExecutionError>(leftVal - rightVal);
  case Mul:
    return Ok<Value, ExecutionError>(leftVal * rightVal);
  case Div:
    // Could add division by zero check here
    return Ok<Value, ExecutionError>(leftVal / rightVal);
  case Mod:
    return Ok<Value, ExecutionError>(leftVal % rightVal);
  case Eq:
    return Ok<Value, ExecutionError>(Value(leftVal == rightVal));
  case Ne:
    return Ok<Value, ExecutionError>(Value(leftVal != rightVal));
  case Lt:
    return Ok<Value, ExecutionError>(Value(leftVal < rightVal));
  case Le:
    return Ok<Value, ExecutionError>(Value(leftVal <= rightVal));
  case Gt:
    return Ok<Value, ExecutionError>(Value(leftVal > rightVal));
  case Ge:
    return Ok<Value, ExecutionError>(Value(leftVal >= rightVal));
  case And:
    return Ok<Value, ExecutionError>(leftVal && rightVal);
  case Or:
    return Ok<Value, ExecutionError>(leftVal || rightVal);
  default:
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
}

Result<Value, ExecutionError> UnaryOpExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                          ThreadgroupContext &tg) const {
  std::cout << "DEBUG: UnaryOpExpr - Lane " << lane.laneId << " evaluating unary operation (Result-based)" << std::endl;
  
  auto exprResult = expr_->evaluate_result(lane, wave, tg);
  if (exprResult.is_err()) {
    return exprResult;
  }
  Value val = exprResult.unwrap();

  switch (op_) {
  case Neg:
  case Minus:
    return Ok<Value, ExecutionError>(Value(-val.asFloat()));
  case Not:
  case LogicalNot:
    return Ok<Value, ExecutionError>(!val);
  case Plus:
    return Ok<Value, ExecutionError>(val);
  case PreIncrement:
    // Note: This modifies the variable, would need variable reference
    // For now, fall back to exception-based approach
    try {
      Value result = evaluate(lane, wave, tg);
      return Ok<Value, ExecutionError>(result);
    } catch (const std::exception &) {
      return Err<Value, ExecutionError>(ExecutionError::InvalidState);
    }
  case PostIncrement:
  case PreDecrement:
  case PostDecrement:
    // These modify variables, need special handling
    try {
      Value result = evaluate(lane, wave, tg);
      return Ok<Value, ExecutionError>(result);
    } catch (const std::exception &) {
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


Result<Value, ExecutionError> LaneIndexExpr::evaluate_result(LaneContext &lane, WaveContext &,
                                                            ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(lane.laneId)));
}

Result<Value, ExecutionError> WaveIndexExpr::evaluate_result(LaneContext &, WaveContext &wave,
                                                            ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(wave.waveId)));
}

Result<Value, ExecutionError> ThreadIndexExpr::evaluate_result(LaneContext &lane, WaveContext &,
                                                              ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(lane.laneId)));
}

Result<Value, ExecutionError> ConditionalExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                              ThreadgroupContext &tg) const {
  auto condResult = condition_->evaluate_result(lane, wave, tg);
  if (condResult.is_err()) {
    return condResult;
  }
  
  if (condResult.unwrap().asInt()) {
    return trueExpr_->evaluate_result(lane, wave, tg);
  } else {
    return falseExpr_->evaluate_result(lane, wave, tg);
  }
}

Result<Value, ExecutionError> WaveGetLaneCountExpr::evaluate_result(LaneContext &, WaveContext &wave,
                                                                   ThreadgroupContext &) const {
  return Ok<Value, ExecutionError>(Value(static_cast<int>(wave.waveSize)));
}

Result<Value, ExecutionError> WaveIsFirstLaneExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                                  ThreadgroupContext &tg) const {
  // Use exception-based approach since this requires complex wave state analysis
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }
}

Result<Value, ExecutionError> WaveActiveAllEqualExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                                     ThreadgroupContext &tg) const {
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }
}

Result<Value, ExecutionError> WaveActiveAllTrueExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                                    ThreadgroupContext &tg) const {
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }
}

Result<Value, ExecutionError> WaveActiveAnyTrueExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                                    ThreadgroupContext &tg) const {
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }
}

Result<Value, ExecutionError> SharedReadExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                             ThreadgroupContext &tg) const {
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::WaveOperationWait);
  }
}

Result<Value, ExecutionError> BufferAccessExpr::evaluate_result(LaneContext &lane, WaveContext &wave,
                                                               ThreadgroupContext &tg) const {
  // BufferAccessExpr has complex buffer lookup logic, fall back to exception-based approach
  try {
    Value result = evaluate(lane, wave, tg);
    return Ok<Value, ExecutionError>(result);
  } catch (const std::exception &) {
    return Err<Value, ExecutionError>(ExecutionError::InvalidState);
  }
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
  std::pair<const void *, uint32_t> instructionKey = {
      static_cast<const void *>(this), currentBlockId};

  std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
            << " executing WaveActiveSum in block " << currentBlockId
            << ", instruction key=(" << static_cast<const void *>(this) << ","
            << currentBlockId << ")" << std::endl;

  // CRITICAL: If lane is resuming from wave operation, check for stored results
  // first
  if (lane.isResumingFromWaveOp) {
    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
              << " is resuming from wave operation, checking for stored result"
              << std::endl;

    auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
    if (syncPointIt != wave.activeSyncPoints.end()) {
      auto &syncPoint = syncPointIt->second;
      if (syncPoint.getPhase() == SyncPointState::Executed) {
        try {
          // Use state machine method to retrieve result
          Value result = syncPoint.retrieveResult(lane.laneId);
          std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
                    << " retrieving stored wave result: " << result.toString()
                    << " (phase: "
                    << syncPointStateToString(syncPoint.getPhase()) << ")"
                    << std::endl;

          // Clear the resuming flag - we successfully retrieved the result
          const_cast<LaneContext &>(lane).isResumingFromWaveOp = false;
          return result;
        } catch (const std::runtime_error &e) {
          std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
                    << " failed to retrieve result: " << e.what() << std::endl;
        }
      }
    }

    // No stored result found - clear flag and continue with normal execution
    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
              << " no stored result found for key (" << instructionKey.first
              << "," << instructionKey.second
              << "), continuing with normal execution" << std::endl;

    // Debug: show what results are actually available
    auto debugSyncPointIt = wave.activeSyncPoints.find(instructionKey);
    if (debugSyncPointIt != wave.activeSyncPoints.end()) {
      std::cout << "DEBUG: WAVE_OP: Available results for lanes: ";
      for (const auto &[availableLaneId, value] :
           debugSyncPointIt->second.pendingResults) {
        std::cout << availableLaneId << " ";
      }
      std::cout << std::endl;
    } else {
      std::cout
          << "DEBUG: WAVE_OP: No sync point found for this instruction key"
          << std::endl;
    }

    const_cast<LaneContext &>(lane).isResumingFromWaveOp = false;
  }

  // Check if there's already a computed result for this lane (normal path)
  auto syncPointIt = wave.activeSyncPoints.find(instructionKey);
  if (syncPointIt != wave.activeSyncPoints.end()) {
    auto &syncPoint = syncPointIt->second;
    if (syncPoint.getPhase() == SyncPointState::Executed) {
      try {
        // Use state machine method to retrieve result
        Value result = syncPoint.retrieveResult(lane.laneId);
        INTERPRETER_DEBUG_LOG(
            "Lane " << lane.laneId
                    << " retrieving stored wave result: " << result.toString()
                    << " (phase: " << (int)syncPoint.getPhase() << ")\n");

        return result;
      } catch (const std::runtime_error &e) {
        INTERPRETER_DEBUG_LOG("Lane " << lane.laneId
                                      << " failed to retrieve result: "
                                      << e.what() << "\n");
      }
    }
  }

  // Mark this lane as waiting at this specific instruction
  tg.markLaneWaitingAtWaveInstruction(wave.waveId, lane.laneId,
                                      static_cast<const void *>(this),
                                      "WaveActiveOp");
  // No stored result - check if we can execute or need to wait
  // In collective execution model, all lanes should wait for
  // processWaveOperations to handle collective execution Only skip waiting if
  // we're resuming from a previous wave operation
  if (!lane.isResumingFromWaveOp) {

    // Store the compound key in the sync point for proper tracking
    auto &syncPoint = wave.activeSyncPoints[instructionKey];
    syncPoint.instruction =
        static_cast<const void *>(this); // Store original instruction pointer

    // Get current block and mark lane as waiting
    tg.markLaneWaitingForWave(wave.waveId, lane.laneId, currentBlockId);

    std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
              << " cannot execute, starting to wait in block " << currentBlockId
              << std::endl;

    // CRITICAL: Force refresh of block resolution status after marking lane as
    // waiting This is essential because the block needs to know all
    // participants are now resolved
    // Resolution status is now tracked by registry - no need for old system
    // metadata
    std::cout
        << "DEBUG: WAVE_OP: Resolution status tracked by registry for block "
        << currentBlockId << std::endl;

    // CRITICAL: 3-step logic for wave operation re-evaluation
    // Step 1: Check if this newly waiting lane completes the participant set
    // Step 2: Re-evaluate if wave operations can now execute with all
    // participants waiting Step 3: If ready, mark sync point as complete so
    // main loop will execute and wake lanes

    if (tg.canExecuteWaveInstruction(wave.waveId, lane.laneId,
                                     static_cast<const void *>(this))) {
      std::cout << "DEBUG: WAVE_OP: After lane " << lane.laneId
                << " started waiting, wave operation can now execute!"
                << std::endl;

      // No need to update flags - they are computed on-demand now

      std::cout << "DEBUG: WAVE_OP: Updated sync point - allParticipantsKnown="
                << syncPoint.isAllParticipantsKnown(tg, wave.waveId)
                << ", allParticipantsArrived="
                << syncPoint.isAllParticipantsArrived() << ", readyToExecute="
                << syncPoint.isReadyToExecute(tg, wave.waveId) << std::endl;
    }

    // Throw a special exception to indicate we need to wait
    throw WaveOperationWaitException();
  }

  // This shouldn't happen in the new collective model, but keep as fallback
  std::cout << "DEBUG: WAVE_OP: Lane " << lane.laneId
            << " hit fallback path for instruction " << instructionKey.first
            << " block " << instructionKey.second << std::endl;
  std::cout << "DEBUG: WAVE_OP: Lane state - isResumingFromWaveOp="
            << lane.isResumingFromWaveOp << std::endl;
  std::cout << "DEBUG: WAVE_OP: Available sync points: "
            << wave.activeSyncPoints.size() << std::endl;
  for (const auto &[key, syncPoint] : wave.activeSyncPoints) {
    std::cout << "DEBUG: WAVE_OP: Sync point (" << key.first << ","
              << key.second << ") has " << syncPoint.pendingResults.size()
              << " pending results" << std::endl;
  }
  throw std::runtime_error(
      "Wave operation fallback path - should not reach here");
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
  } catch (const WaveOperationWaitException &) {
    // Lane is waiting for wave operation - re-throw so control flow statements
    // can handle it
    std::cout << "DEBUG: VarDeclStmt - Lane " << lane.laneId
              << " caught WaveOperationWaitException, re-throwing" << std::endl;
    throw; // Re-throw the exception
  }
}

Result<Unit, ExecutionError> VarDeclStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});
  
  // Pure Result-based implementation - no exceptions!
  Value initVal;
  if (init_) {
    initVal = TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  } else {
    initVal = Value(0);
  }
  
  lane.variables[name_] = initVal;
  return Ok<Unit, ExecutionError>(Unit{});
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
  } catch (const WaveOperationWaitException &) {
    // Lane is waiting for wave operation - re-throw so control flow statements
    // can handle it
    std::cout << "DEBUG: AssignStmt - Lane " << lane.laneId
              << " caught WaveOperationWaitException, re-throwing" << std::endl;
    throw; // Re-throw the exception
  }
}

Result<Unit, ExecutionError> AssignStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                      ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});
  
  // Pure Result-based implementation - no exceptions!
  Value val = TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  lane.variables[name_] = val;
  return Ok<Unit, ExecutionError>(Unit{});
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
  int ourStackIndex = findStackIndex(lane);

  bool isResuming = (ourStackIndex >= 0);

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " starting fresh execution (pushed to stack depth="
              << lane.executionStack.size() << ", this=" << this << ")"
              << std::endl;
  } else {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " resuming execution (found at stack index=" << ourStackIndex
              << ", current stack depth=" << lane.executionStack.size()
              << ", this=" << this << ")" << std::endl;
  }

  // Don't hold reference to vector element - it can be invalidated during
  // nested execution
  bool hasElse = !elseBlock_.empty();
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  // Check if this lane has already set up blocks in its execution stack
  auto &ourEntry = lane.executionStack[ourStackIndex];
  bool setupComplete =
      (ourEntry.ifThenBlockId != 0 || ourEntry.ifElseBlockId != 0 ||
       ourEntry.ifMergeBlockId != 0);

  try {
    // while (lane.isActive) {
    // Use our entry, not back() - already declared above
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId << " in phase "
              << LaneContext::getPhaseString(ourEntry.phase)
              << " (stack depth=" << lane.executionStack.size()
              << ", our index=" << ourStackIndex << ", this=" << this << ")"
              << std::endl;
    switch (ourEntry.phase) {

    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      evaluateConditionAndSetup(lane, wave, tg, ourStackIndex, parentBlockId, hasElse);
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::ExecutingThenBlock: {
      executeThenBranch(lane, wave, tg, ourStackIndex);
      if (lane.hasReturned) {
        return;
      }
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::ExecutingElseBlock: {
      executeElseBranch(lane, wave, tg, ourStackIndex);
      if (lane.hasReturned) {
        return;
      }
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      performReconvergence(lane, wave, tg, ourStackIndex, hasElse);
      return;
    }
    }
    // }

  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - execution state is already saved
    // Note: ourEntry might be out of scope here, so we need to access via
    // ourStackIndex
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " waiting for wave operation in phase "
              << (int)lane.executionStack[ourStackIndex].phase
              << " at statement "
              << lane.executionStack[ourStackIndex].statementIndex << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    // Propagate break/continue to enclosing loop - do NOT move to merge block
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " popping stack due to ControlFlowException (depth "
              << lane.executionStack.size() << "->"
              << (lane.executionStack.size() - 1) << ", this=" << this << ")"
              << std::endl;

    // Get block IDs before popping the stack
    uint32_t ifThenBlockId = lane.executionStack[ourStackIndex].ifThenBlockId;
    uint32_t ifElseBlockId = lane.executionStack[ourStackIndex].ifElseBlockId;
    uint32_t ifMergeBlockId = lane.executionStack[ourStackIndex].ifMergeBlockId;

    lane.executionStack.pop_back();
    tg.popMergePoint(wave.waveId, lane.laneId);

    // Clean up then/else blocks - lane will never return to them
    tg.removeThreadFromAllSets(ifThenBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(ifThenBlockId, wave.waveId, lane.laneId);

    if (hasElse && ifElseBlockId != 0) {
      tg.removeThreadFromAllSets(ifElseBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(ifElseBlockId, wave.waveId, lane.laneId);
    }

    // Also clean up merge block since we're not going there
    tg.removeThreadFromAllSets(ifMergeBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(ifMergeBlockId, wave.waveId, lane.laneId);

    // Restore active state (reconvergence)
    lane.isActive = lane.isActive && !lane.hasReturned;
    throw;
  }

  // This appears to be unreachable code after the exception handling
  // If it's needed, it should use execution stack values but this looks like
  // dead code

  // Restore active state (reconvergence)
  lane.isActive = lane.isActive && !lane.hasReturned;
}

// Helper methods for IfStmt execute phases
void IfStmt::evaluateConditionAndSetup(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                int ourStackIndex, uint32_t parentBlockId, bool hasElse) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " evaluating condition" << std::endl;

  // Only evaluate condition if not already evaluated (avoid re-evaluation on resume)
  if (!ourEntry.conditionEvaluated) {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " evaluating condition for first time" << std::endl;
    // Evaluate condition (can throw WaveOperationWaitException)
    bool conditionResult = condition_->evaluate(lane, wave, tg).asBool();
    lane.executionStack[ourStackIndex].conditionResult = conditionResult;
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " condition result=" << ourEntry.conditionResult
              << std::endl;
  } else {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " using cached condition result="
              << ourEntry.conditionResult << std::endl;
  }

  // Condition evaluated successfully - set up blocks
  std::set<uint32_t> divergentBlocks;
  tg.pushMergePoint(wave.waveId, lane.laneId,
                    static_cast<const void *>(this), parentBlockId,
                    divergentBlocks);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  auto blockIds =
      tg.createIfBlocks(static_cast<const void *>(this), parentBlockId,
                        currentMergeStack, hasElse, lane.executionPath);
  ourEntry.ifThenBlockId = std::get<0>(blockIds);
  ourEntry.ifElseBlockId = std::get<1>(blockIds);
  ourEntry.ifMergeBlockId = std::get<2>(blockIds);
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " setup complete: thenBlockId=" << ourEntry.ifThenBlockId
            << ", elseBlockId=" << ourEntry.ifElseBlockId
            << ", mergeBlockId=" << ourEntry.ifMergeBlockId << std::endl;

  // Update blocks based on condition result
  if (ourEntry.conditionResult) {
    tg.moveThreadFromUnknownToParticipating(ourEntry.ifThenBlockId,
                                            wave.waveId, lane.laneId);
    if (hasElse) {
      tg.removeThreadFromUnknown(ourEntry.ifElseBlockId, wave.waveId,
                                 lane.laneId);
      tg.removeThreadFromNestedBlocks(ourEntry.ifElseBlockId, wave.waveId,
                                      lane.laneId);
    }
    // Don't remove from merge block yet - lane will reconverge there later

    ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingThenBlock;
    ourEntry.inThenBranch = true;
    ourEntry.blockId = ourEntry.ifThenBlockId;
  } else if (hasElse) {
    tg.moveThreadFromUnknownToParticipating(ourEntry.ifElseBlockId,
                                            wave.waveId, lane.laneId);
    tg.removeThreadFromUnknown(ourEntry.ifThenBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(ourEntry.ifThenBlockId, wave.waveId,
                                    lane.laneId);
    // Don't remove from merge block yet - lane will reconverge there later

    ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingElseBlock;
    ourEntry.inThenBranch = false;
    ourEntry.blockId = ourEntry.ifElseBlockId;
  } else {
    // No else block
    tg.removeThreadFromUnknown(ourEntry.ifThenBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(ourEntry.ifThenBlockId, wave.waveId,
                                    lane.laneId);
    tg.moveThreadFromUnknownToParticipating(ourEntry.ifMergeBlockId,
                                            wave.waveId, lane.laneId);

    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
  }

  lane.executionStack[ourStackIndex].statementIndex = 0;
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " condition=" << ourEntry.conditionResult
            << ", moving to phase="
            << LaneContext::getPhaseString(ourEntry.phase) << std::endl;
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " executing: thenBlockId=" << ourEntry.ifThenBlockId
            << ", elseBlockId=" << ourEntry.ifElseBlockId
            << ", mergeBlockId=" << ourEntry.ifMergeBlockId << std::endl;
}

void IfStmt::executeThenBranch(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                       int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " executing then block from statement "
            << ourEntry.statementIndex << std::endl;
  // Execute statements in then block from saved position
  for (size_t i = ourEntry.statementIndex; i < thenBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    thenBlock_[i]->execute(lane, wave, tg);
    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return;
    }
    // TODO: additional cleanup?
    if (lane.hasReturned) {
      std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ")" << std::endl;
      // TODO: verify if need additional clenaup
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  // Completed then block
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " completed then block, moving to reconvergence"
            << std::endl;
}

void IfStmt::executeElseBranch(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                       int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " executing else block from statement "
            << ourEntry.statementIndex << std::endl;

  // Execute statements in else block from saved position
  for (size_t i = ourEntry.statementIndex; i < elseBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    elseBlock_[i]->execute(lane, wave, tg);
    // TODO: verify if need additional cleanup
    if (lane.hasReturned) {
      std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ")" << std::endl;
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return;
    }

    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }

  // Completed else block
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " completed else block, moving to reconvergence"
            << std::endl;
}

void IfStmt::performReconvergence(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                          int ourStackIndex, bool hasElse) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  uint32_t currentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  uint32_t laneSpecificMergeBlockId = ourEntry.ifMergeBlockId;
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " performing reconvergence from block " << currentBlockId
            << " to laneSpecificMergeBlockId=" << laneSpecificMergeBlockId
            << std::endl;

  // Debug: Show current merge stack before popping
  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " merge stack before reconvergence (size="
            << currentMergeStack.size() << "):" << std::endl;
  for (size_t i = 0; i < currentMergeStack.size(); i++) {
    std::cout << "  Stack[" << i << "]: sourceStatement="
              << currentMergeStack[i].sourceStatement << std::endl;
  }

  // Use stored merge block ID - don't recreate blocks during reconvergence
  // mergeBlockId should already be set from initial setup

  // Clean up execution state
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " popping stack at reconvergence (depth "
            << lane.executionStack.size() << "->"
            << (lane.executionStack.size() - 1) << ", this=" << this << ")"
            << std::endl;
  lane.executionStack.pop_back();

  // Reconverge at merge block
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " popping merge point before assigning to block "
            << laneSpecificMergeBlockId << std::endl;
  tg.popMergePoint(wave.waveId, lane.laneId);
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " assigning to merge block " << laneSpecificMergeBlockId
            << std::endl;
  // tg.assignLaneToBlock(wave.waveId, lane.laneId,
  // laneSpecificMergeBlockId);

  // Move lane to merge block as participating (reconvergence)
  tg.moveThreadFromUnknownToParticipating(laneSpecificMergeBlockId,
                                          wave.waveId, lane.laneId);

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

  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " reconvergence complete" << std::endl;
}

Result<Unit, ExecutionError> IfStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                  ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based implementation for basic if statements
  // For now, implement a simplified version that evaluates condition and executes branches
  
  // Evaluate condition using Result-based approach
  auto conditionResult = condition_->evaluate_result(lane, wave, tg);
  if (conditionResult.is_err()) {
    return Err<Unit, ExecutionError>(conditionResult.unwrap_err());
  }
  
  bool condValue = conditionResult.unwrap().asBool();
  
  // Execute appropriate branch
  if (condValue && !thenBlock_.empty()) {
    // Execute then block statements
    for (const auto &stmt : thenBlock_) {
      auto result = stmt->execute_result(lane, wave, tg);
      if (result.is_err()) {
        return result; // Propagate error (including control flow errors)
      }
    }
  } else if (!condValue && !elseBlock_.empty()) {
    // Execute else block statements  
    for (const auto &stmt : elseBlock_) {
      auto result = stmt->execute_result(lane, wave, tg);
      if (result.is_err()) {
        return result; // Propagate error (including control flow errors)
      }
    }
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
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

// Result-based versions of IfStmt helper methods
Result<Unit, ExecutionError> IfStmt::evaluateConditionAndSetup_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                int ourStackIndex, uint32_t parentBlockId, bool hasElse) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " evaluating condition (Result-based)" << std::endl;

  // Only evaluate condition if not already evaluated (avoid re-evaluation on resume)
  if (!ourEntry.conditionEvaluated) {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " evaluating condition for first time (Result-based)" << std::endl;
    
    // Evaluate condition using Result-based evaluation
    Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
    bool conditionResult = condVal.asBool();
    
    lane.executionStack[ourStackIndex].conditionResult = conditionResult;
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " condition result=" << ourEntry.conditionResult
              << " (Result-based)" << std::endl;
  } else {
    std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
              << " using cached condition result="
              << ourEntry.conditionResult << " (Result-based)" << std::endl;
  }

  // Condition evaluated successfully - set up blocks
  std::set<uint32_t> divergentBlocks;
  tg.pushMergePoint(wave.waveId, lane.laneId,
                    static_cast<const void *>(this), parentBlockId,
                    divergentBlocks);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  auto blockIds =
      tg.createIfBlocks(static_cast<const void *>(this), parentBlockId,
                        currentMergeStack, hasElse, lane.executionPath);
  ourEntry.ifThenBlockId = std::get<0>(blockIds);
  ourEntry.ifElseBlockId = std::get<1>(blockIds);
  ourEntry.ifMergeBlockId = std::get<2>(blockIds);
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " setup complete: thenBlockId=" << ourEntry.ifThenBlockId
            << ", elseBlockId=" << ourEntry.ifElseBlockId 
            << ", mergeBlockId=" << ourEntry.ifMergeBlockId
            << " (Result-based)" << std::endl;

  // Choose which branch to execute based on condition
  if (ourEntry.conditionResult) {
    ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingThenBlock;
    ourEntry.statementIndex = 0;
  } else {
    if (hasElse) {
      ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingElseBlock;
      ourEntry.statementIndex = 0;
    } else {
      ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    }
  }

  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> IfStmt::executeThenBranch_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                       int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " executing then block from statement "
            << ourEntry.statementIndex << " (Result-based)" << std::endl;
  
  // Execute statements in then block from saved position
  for (size_t i = ourEntry.statementIndex; i < thenBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    
    // Use Result-based execute_result instead of exception-based execute
    TRY_RESULT(thenBlock_[i]->execute_result(lane, wave, tg), Unit, ExecutionError);
    
    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return Ok<Unit, ExecutionError>(Unit{});
    }
    
    if (lane.hasReturned) {
      std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ") (Result-based)" << std::endl;
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> IfStmt::executeElseBranch_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                       int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
            << " executing else block from statement "
            << ourEntry.statementIndex << " (Result-based)" << std::endl;

  // Execute statements in else block from saved position
  for (size_t i = ourEntry.statementIndex; i < elseBlock_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    
    // Use Result-based execute_result instead of exception-based execute
    TRY_RESULT(elseBlock_[i]->execute_result(lane, wave, tg), Unit, ExecutionError);
    
    if (lane.hasReturned) {
      std::cout << "DEBUG: IfStmt - Lane " << lane.laneId
                << " popping stack due to return (depth "
                << lane.executionStack.size() << "->"
                << (lane.executionStack.size() - 1) << ", this=" << this
                << ") (Result-based)" << std::endl;
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
  
  return Ok<Unit, ExecutionError>(Unit{});
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
  int ourStackIndex = findStackIndex(lane);

  bool isResuming = (ourStackIndex >= 0);
  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;
  uint32_t parentBlockId = 0;

  if (!isResuming) {
    // Starting fresh - push initial state for initialization
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingInit);
    ourStackIndex = lane.executionStack.size() - 1;
    
    // Set up fresh execution using extracted helper method
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId, mergeBlockId);
  } else {
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
              << " resuming execution (found at stack index=" << ourStackIndex
              << ", current stack depth=" << lane.executionStack.size()
              << ", this=" << this << ")" << std::endl;

    // Restore saved block IDs
    headerBlockId = lane.executionStack[ourStackIndex].loopHeaderBlockId;
    mergeBlockId = lane.executionStack[ourStackIndex].loopMergeBlockId;
  }

  // Execute loop with state machine
  try {
    auto &ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " in phase "
              << LaneContext::getPhaseString(ourEntry.phase)
              << " (stack depth=" << lane.executionStack.size()
              << ", our index=" << ourStackIndex << ", this=" << this << ")"
              << std::endl;

    switch (ourEntry.phase) {
    case LaneContext::ControlFlowPhase::EvaluatingInit: {
      // Evaluate initialization using extracted helper method
      evaluateInitPhase(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      // Evaluate condition using extracted helper method
      evaluateConditionPhase(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::ExecutingBody: {
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
                << " executing body for iteration " << ourEntry.loopIteration
                << " from statement " << ourEntry.statementIndex << std::endl;

      // Set up iteration-specific blocks using extracted helper method
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);

      // Execute body statements using extracted helper method
      executeBodyStatements(lane, wave, tg, ourStackIndex, headerBlockId);
      
      // Check if we need to return early (lane returned or needs resume)
      if (lane.hasReturned || lane.state != ThreadState::Ready) {
        return;
      }

      // Clean up after body execution using extracted helper method
      cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::EvaluatingIncrement: {
      // Evaluate increment using extracted helper method
      evaluateIncrementPhase(lane, wave, tg, ourStackIndex);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      // Handle loop exit using extracted helper method
      handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
      return;
    }

    default:
      std::cout << "ERROR: ForStmt - Unexpected phase "
                << static_cast<int>(ourEntry.phase) << std::endl;
      return;
    }
    // }
  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - execution state is already saved
    auto &ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
              << " waiting for wave operation in phase "
              << static_cast<int>(ourEntry.phase) << " at statement "
              << ourEntry.statementIndex << ", iteration "
              << ourEntry.loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }
  }
}

// Pure Result-based ForStmt phase implementations
Result<Unit, ExecutionError> ForStmt::executeInit(LaneContext &lane, WaveContext &wave,
                                                 ThreadgroupContext &tg) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing init (Result-based)" << std::endl;
  
  // Initialize loop variable using Result-based evaluation
  Value initVal = TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  lane.variables[loopVar_] = initVal;
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<bool, ExecutionError> ForStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating condition (Result-based)" << std::endl;
  
  // Evaluate condition using Result-based evaluation
  Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), bool, ExecutionError);
  bool shouldContinue = condVal.asBool();
  
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError> ForStmt::executeBody(LaneContext &lane, WaveContext &wave,
                                                 ThreadgroupContext &tg, size_t &statementIndex) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing body (Result-based)" << std::endl;
  
  // Execute body statements using Result-based approach
  for (size_t i = statementIndex; i < body_.size(); ++i) {
    auto result = body_[i]->execute_result(lane, wave, tg);
    if (result.is_err()) {
      // Handle control flow errors
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: ForStmt - Break encountered in body" << std::endl;
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: ForStmt - Continue encountered in body" << std::endl;
        statementIndex = body_.size(); // Skip remaining statements
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

Result<Unit, ExecutionError> ForStmt::executeIncrement(LaneContext &lane, WaveContext &wave,
                                                     ThreadgroupContext &tg) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing increment (Result-based)" << std::endl;
  
  // Execute increment expression using Result-based evaluation
  TRY_RESULT(increment_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                   ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing pure Result-based for loop" << std::endl;

  // Pure Result-based for loop implementation (simplified, no complex state management)
  // This is a basic implementation - for full compatibility with the complex phase-based
  // execution, we'd need to implement the full state machine with Results
  
  // Phase 1: Initialize
  auto initResult = executeInit(lane, wave, tg);
  if (initResult.is_err()) {
    return initResult;
  }
  
  // Phase 2: Loop with condition checking
  size_t statementIndex = 0;
  while (true) {
    // Evaluate condition
    auto condResult = evaluateCondition(lane, wave, tg);
    if (condResult.is_err()) {
      return Err<Unit, ExecutionError>(condResult.unwrap_err());
    }
    
    bool shouldContinue = condResult.unwrap();
    if (!shouldContinue) {
      std::cout << "DEBUG: ForStmt - Loop condition false, exiting" << std::endl;
      break; // Exit loop
    }
    
    // Execute body
    auto bodyResult = executeBody(lane, wave, tg, statementIndex);
    if (bodyResult.is_err()) {
      ExecutionError error = bodyResult.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: ForStmt - Breaking from loop" << std::endl;
        break; // Exit loop
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: ForStmt - Continuing loop" << std::endl;
        // Continue to increment phase
      } else {
        // Other errors (like WaveOperationWait) should be propagated
        return bodyResult;
      }
    }
    
    // Execute increment
    auto incResult = executeIncrement(lane, wave, tg);
    if (incResult.is_err()) {
      return incResult;
    }
  }
  
  std::cout << "DEBUG: ForStmt - Loop completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

// Helper method for executing body statements in ForStmt
void ForStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                   int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements - start in iteration block, naturally flow to merge blocks
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing statement " << i 
              << " in block " << blockBeforeStatement << std::endl;

    body_[i]->execute(lane, wave, tg);

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() && currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId 
                  << " popped iteration merge point on early return" << std::endl;
      }
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " child statement needs resume" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement 
                << ", Block after: " << blockAfterStatement << std::endl;
      return;
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement " << i << " (likely merge block)" << std::endl;
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
}

// Helper method for setting up iteration-specific blocks in ForStmt
void ForStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                  int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Check if we need an iteration-specific starting block
  // Only create one if we have non-control-flow statements before any control flow
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      // We have regular statements that need unique iteration context
      needsIterationBlock = true;
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
                << " needs iteration block (first statement is not control flow)" << std::endl;
    } else {
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
                << " no iteration block needed (first statement is control flow)" << std::endl;
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block (unique context for this iteration)
    iterationStartBlockId = ourEntry.loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      // Create unique identity for this iteration's starting block
      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x3000);

      // Use REGULAR block type - this is the starting context for this iteration
      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId, currentMergeStack, true, lane.executionPath);

      // Try to find existing block first
      iterationStartBlockId = tg.findBlockByIdentity(iterationIdentity);

      if (iterationStartBlockId == 0) {
        // Create new iteration starting block
        std::map<WaveId, std::set<LaneId>> expectedLanes = tg.getCurrentBlockParticipants(headerBlockId);
        iterationStartBlockId = tg.findOrCreateBlockForPath(iterationIdentity, expectedLanes);
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " created iteration starting block "
                  << iterationStartBlockId << " for iteration " << ourEntry.loopIteration << std::endl;
      } else {
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " found existing iteration starting block "
                  << iterationStartBlockId << " for iteration " << ourEntry.loopIteration << std::endl;
      }

      ourEntry.loopBodyBlockId = iterationStartBlockId;
    }

    // Move to iteration-specific block
    if (tg.getCurrentBlock(wave.waveId, lane.laneId) != iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId, wave.waveId, lane.laneId);
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " moved to iteration starting block "
                << iterationStartBlockId << std::endl;
    }
  } else {
    // No iteration block needed - but we still need to ensure unique execution context per iteration
    // Only push merge point if we're at the beginning of the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " merge stack size: " << currentMergeStack.size() << std::endl;
      for (size_t i = 0; i < currentMergeStack.size(); i++) {
        std::cout << "  Stack[" << i << "]: sourceStatement=" << currentMergeStack[i].sourceStatement << std::endl;
      }
      std::cout << "  Looking for iterationMarker=" << iterationMarker << std::endl;
      bool alreadyPushed = hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack" << std::endl;
      }

      if (!alreadyPushed) {
        std::set<uint32_t> emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId), emptyDivergentBlocks);
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " pushed iteration merge point "
                  << iterationMarker << " (no iteration block needed, but merge stack modified)" << std::endl;
      }
    }
  }
}

// Helper method for body completion cleanup in ForStmt
void ForStmt::cleanupAfterBodyExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                       int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Body completed - clean up iteration-specific merge point
  const void *iterationMarker = reinterpret_cast<const void *>(
      reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

  std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration " << ourEntry.loopIteration << std::endl;
  }

  // Move back to header block for increment phase
  uint32_t finalBlock = tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " body completed in block " << finalBlock
            << ", moving to header block " << headerBlockId
            << " for increment" << std::endl;
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  ourEntry.phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for increment evaluation phase in ForStmt
void ForStmt::evaluateIncrementPhase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                    int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating increment for iteration " << ourEntry.loopIteration << std::endl;

  // Increment loop variable (side effect, don't assign result)
  increment_->evaluate(lane, wave, tg);

  // Move to next iteration
  lane.executionStack[ourStackIndex].loopIteration++;
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for loop exit/reconverging phase in ForStmt
void ForStmt::handleLoopExit(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                            int ourStackIndex, uint32_t mergeBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " exiting loop after " << ourEntry.loopIteration << " iterations" << std::endl;

  // Clean up execution state
  lane.executionStack.pop_back();

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);
}

// Helper method for break exception handling in ForStmt
void ForStmt::handleBreakException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                  int ourStackIndex, uint32_t headerBlockId) {
  // Break - exit loop
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " breaking from loop" << std::endl;
  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for continue exception handling in ForStmt
void ForStmt::handleContinueException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                     int ourStackIndex, uint32_t headerBlockId) {
  // Continue - go to increment phase and skip remaining statements
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " continuing loop" << std::endl;

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId, ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    std::cout << "DEBUG: ForStmt - Marked lane " << lane.laneId << " as Left from block "
              << ourEntry.loopBodyBlockId << std::endl;
  }

  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from all nested blocks this lane is abandoning
  if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
    tg.removeThreadFromAllSets(lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(lane.executionStack[ourStackIndex].loopBodyBlockId,
                                    wave.waveId, lane.laneId);
  }
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].loopBodyBlockId = 0; // Reset for next iteration
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;

  // Set state to WaitingForResume to prevent currentStatement increment
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for fresh execution setup in ForStmt
void ForStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                 int ourStackIndex, uint32_t &headerBlockId, uint32_t &mergeBlockId) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " starting fresh execution (pushed to stack depth="
            << lane.executionStack.size() << ", this=" << this << ")" << std::endl;

  // Get current block before entering loop
  uint32_t parentBlockId = tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);

  // Get current merge stack for block creation
  std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);

  // Create loop blocks (header, merge) - pass current execution path
  auto [hBlockId, mBlockId] = tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                                                  currentMergeStack, lane.executionPath);
  headerBlockId = hBlockId;
  mergeBlockId = mBlockId;

  // Save block IDs in our stack entry
  lane.executionStack[ourStackIndex].loopHeaderBlockId = headerBlockId;
  lane.executionStack[ourStackIndex].loopMergeBlockId = mergeBlockId;

  // Push merge point for loop divergence
  std::set<uint32_t> divergentBlocks = {headerBlockId};
  tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this), parentBlockId,
                    divergentBlocks);
}

// Helper method for initialization phase in ForStmt
void ForStmt::evaluateInitPhase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                               int ourStackIndex, uint32_t headerBlockId) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating init" << std::endl;

  // Initialize loop variable
  lane.variables[loopVar_] = init_->evaluate(lane, wave, tg);

  // Move to loop header block
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);

  // Move to condition evaluation phase
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for condition evaluation phase in ForStmt
void ForStmt::evaluateConditionPhase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration << std::endl;

  // Check loop condition
  bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return;
  }

  // Condition passed, move to body execution
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Result-based versions of ForStmt helper methods
Result<Unit, ExecutionError> ForStmt::executeBodyStatements_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                   int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements - start in iteration block, naturally flow to merge blocks
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing statement " << i 
              << " in block " << blockBeforeStatement << " (Result-based)" << std::endl;

    // Use Result-based execute_result instead of exception-based execute
    TRY_RESULT(body_[i]->execute_result(lane, wave, tg), Unit, ExecutionError);

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() && currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId 
                  << " popped iteration merge point on early return (Result-based)" << std::endl;
      }
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " child statement needs resume (Result-based)" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement 
                << ", Block after: " << blockAfterStatement << std::endl;
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement " << i << " (likely merge block, Result-based)" << std::endl;
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::evaluateInitPhase_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                               int ourStackIndex, uint32_t headerBlockId) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating init (Result-based)" << std::endl;

  // Initialize loop variable using Result-based evaluation
  Value initVal = TRY_RESULT(init_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  lane.variables[loopVar_] = initVal;

  // Move to loop header block
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);

  // Move to condition evaluation phase
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::evaluateConditionPhase_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration << " (Result-based)" << std::endl;

  // Check loop condition using Result-based evaluation
  Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  bool shouldContinue = condVal.asBool();
  
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return Ok<Unit, ExecutionError>(Unit{});
  }

  // Condition passed, move to body execution
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> ForStmt::evaluateIncrementPhase_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                             int ourStackIndex) {
  std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " evaluating increment (Result-based)" << std::endl;

  // Evaluate increment expression using Result-based evaluation  
  Value incrementVal = TRY_RESULT(increment_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  lane.variables[loopVar_] = incrementVal;

  // Move back to condition evaluation
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration++;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
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
  std::cout << "DEBUG: updateWaveOperationStates - Lane " << returningLaneId
            << " returning, checking " << wave.activeSyncPoints.size()
            << " active sync points" << std::endl;

  std::vector<std::pair<const void *, uint32_t>>
      completedInstructions; // Track instructions that become complete

  for (auto &[instructionKey, syncPoint] : wave.activeSyncPoints) {
    bool wasExpected =
        syncPoint.expectedParticipants.count(returningLaneId) > 0;
    std::cout << "DEBUG: updateWaveOperationStates - Sync point "
              << instructionKey.first << " in block " << instructionKey.second
              << ": wasExpected=" << wasExpected << ", allParticipantsKnown="
              << syncPoint.isAllParticipantsKnown(tg, wave.waveId) << std::endl;

    syncPoint.expectedParticipants.erase(returningLaneId);
    syncPoint.arrivedParticipants.erase(returningLaneId);

    // If no participants left, mark this sync point for removal
    if (syncPoint.expectedParticipants.empty()) {
      std::cout << "DEBUG: updateWaveOperationStates - Sync point "
                << instructionKey.first << " in block " << instructionKey.second
                << " has no participants left, marking for removal"
                << std::endl;
      completedInstructions.push_back(instructionKey);
      continue; // Skip further processing for this sync point
    }

    // Check if sync point is now ready for cleanup (has pending results)
    bool isNowReadyForCleanup = syncPoint.isReadyForCleanup();

    std::cout << "DEBUG: updateWaveOperationStates - Lane " << returningLaneId
              << " returning: Block " << syncPoint.blockId
              << " isAllParticipantsKnown="
              << syncPoint.isAllParticipantsKnown(tg, wave.waveId)
              << " for instruction " << instructionKey.first << std::endl;

    // If sync point is ready for cleanup, mark it for processing
    if (isNowReadyForCleanup) {
      std::cout
          << "DEBUG: updateWaveOperationStates - Sync point for instruction "
          << instructionKey.first << " in block " << syncPoint.blockId
          << " became complete due to lane " << returningLaneId << " returning"
          << std::endl;
      completedInstructions.push_back(instructionKey);
    }
  }

  // Wake up lanes waiting at newly completed sync points
  for (const auto &instructionKey : completedInstructions) {
    auto &syncPoint = wave.activeSyncPoints[instructionKey];
    std::cout << "DEBUG: updateWaveOperationStates - Waking up lanes waiting "
                 "at instruction "
              << instructionKey.first << " in block " << instructionKey.second
              << std::endl;
    for (LaneId waitingLaneId : syncPoint.arrivedParticipants) {
      if (waitingLaneId < wave.lanes.size() && wave.lanes[waitingLaneId] &&
          wave.lanes[waitingLaneId]->state == ThreadState::WaitingForWave) {
        std::cout << "DEBUG: updateWaveOperationStates - Waking up lane "
                  << waitingLaneId << " from WaitingForWave to Ready"
                  << std::endl;
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

void BarrierStmt::execute(LaneContext &lane, WaveContext &wave,
                          ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

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

Result<Unit, ExecutionError> ExprStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                    ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  // Pure Result-based implementation - no exceptions!
  if (expr_) {
    // Execute the expression (evaluate it but don't store the result)
    TRY_RESULT(expr_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
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

// Result-based implementations for missing statement types

Result<Unit, ExecutionError> ReturnStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: ReturnStmt - Lane " << lane.laneId << " executing return (Result-based)" << std::endl;
  
  if (expr_) {
    auto exprResult = expr_->evaluate_result(lane, wave, tg);
    if (exprResult.is_err()) {
      return Err<Unit, ExecutionError>(exprResult.unwrap_err());
    }
    lane.returnValue = exprResult.unwrap();
  }

  // Handle comprehensive global cleanup for early return
  handleGlobalEarlyReturn(lane, wave, tg);
  
  std::cout << "DEBUG: ReturnStmt - Return completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> BarrierStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                        ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: BarrierStmt - Lane " << lane.laneId << " executing barrier (Result-based)" << std::endl;
  
  // For now, fall back to exception-based implementation
  // A full Result-based barrier implementation would be complex and require
  // reworking the barrier synchronization logic
  try {
    execute(lane, wave, tg);
    std::cout << "DEBUG: BarrierStmt - Barrier completed successfully" << std::endl;
    return Ok<Unit, ExecutionError>(Unit{});
  } catch (const std::exception &) {
    return Err<Unit, ExecutionError>(ExecutionError::InvalidState);
  }
}

Result<Unit, ExecutionError> SharedWriteStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: SharedWriteStmt - Lane " << lane.laneId << " executing shared write (Result-based)" << std::endl;
  
  auto valueResult = expr_->evaluate_result(lane, wave, tg);
  if (valueResult.is_err()) {
    return Err<Unit, ExecutionError>(valueResult.unwrap_err());
  }
  
  Value value = valueResult.unwrap();
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  tg.sharedMemory->write(addr_, value, tid);
  
  std::cout << "DEBUG: SharedWriteStmt - Shared write completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

Value SharedReadExpr::evaluate(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg) const {
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  return tg.sharedMemory->read(addr_, tid);
}

std::string SharedReadExpr::toString() const {
  return "g_shared[" + std::to_string(addr_) + "]";
}

Value BufferAccessExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                 ThreadgroupContext &tg) const {
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
ExecutionResult MiniHLSLInterpreter::executeWithOrdering(
    const Program &program, const ThreadOrdering &ordering, uint32_t waveSize) {
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
    std::cout << "lane " << lane.laneId << " have completed execution\n";
    lane.state = ThreadState::Completed;
    return true;
  }

  // Execute the current statement using stable exception-based approach
  // Note: All Result-based infrastructure (execute_result methods) remains available
  // This approach maintains compatibility with the existing wave operation system
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
  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - do nothing, statement will be retried
    // Lane state should already be set to WaitingForWave
    std::cout << "DEBUG: WAVE_WAIT: Lane " << (tid % 32)
              << " caught WaveOperationWaitException, state=" << (int)lane.state
              << std::endl;
  } catch (const std::exception &e) {
    lane.state = ThreadState::Error;
    lane.errorMessage = e.what();
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
        const auto* block = tgContext.getBlock(blockId);
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
            std::cout << "DEBUG: WAVE_OP: Updated participants before "
                         "execution to include all current block lanes: ";
            for (LaneId laneId : syncPoint.arrivedParticipants) {
              std::cout << laneId << " ";
            }
            std::cout << std::endl;
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
        std::cout << "DEBUG: processWaveOperations - Sync point "
                  << instructionKey.first << " in block "
                  << instructionKey.second
                  << " has no participants, cleaning up" << std::endl;
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

void MiniHLSLInterpreter::executeCollectiveWaveOperation(
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
      Value value = waveOp->getExpression()->evaluate(*wave.lanes[laneId], wave,
                                                      tgContext);
      values.push_back(value);
    }
  }

  // Execute the wave operation on the collected values
  Value result = waveOp->computeWaveOperation(values);

  // Store the result for all participating lanes and transition to Executed
  // state
  std::cout << "DEBUG: WAVE_OP: Storing collective result for lanes: ";
  for (LaneId laneId : syncPoint.arrivedParticipants) {
    std::cout << laneId << " ";
    syncPoint.pendingResults[laneId] = result;
  }
  // Mark sync point as executed using state machine
  syncPoint.markExecuted();
  std::cout << " (phase: " << (int)syncPoint.getPhase() << ")" << std::endl;

  INTERPRETER_DEBUG_LOG(
      "Collective wave operation result: " << result.toString() << "\n");

  // Transition sync point state to Executed
  syncPoint.state = SyncPointState::Executed;
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertCompoundAssignOperator(
    const clang::CompoundAssignOperator *compoundOp,
    clang::ASTContext &context) {
  std::cout << "Converting compound assignment operator" << std::endl;

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
  auto binaryExpr =
      std::make_unique<BinaryOpExpr>(std::move(lhs), std::move(rhs), opType);

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
            std::cout << "DEBUG: Switch parsing - LHS type: "
                      << lhs->getStmtClassName() << std::endl;
            // Handle implicit casts
            auto unwrapped = lhs->IgnoreImpCasts();
            if (auto intLit =
                    clang::dyn_cast<clang::IntegerLiteral>(unwrapped)) {
              currentCaseValue = intLit->getValue().getSExtValue();
              std::cout << "DEBUG: Switch parsing - found case "
                        << currentCaseValue.value() << std::endl;
            } else {
              std::cout << "DEBUG: Switch parsing - unwrapped LHS type: "
                        << unwrapped->getStmtClassName() << std::endl;
            }
          } else {
            std::cout << "DEBUG: Switch parsing - no LHS found" << std::endl;
          }

          // Handle nested case statements (e.g., case 2: case 3: stmt)
          auto substmt = caseStmt->getSubStmt();
          while (substmt) {
            if (auto nestedCase = clang::dyn_cast<clang::CaseStmt>(substmt)) {
              // This is a nested case - save current case as empty
              if (currentCaseValue.has_value()) {
                std::cout << "DEBUG: Switch parsing - saving empty case "
                          << currentCaseValue.value()
                          << " (falls through to next)" << std::endl;
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
                  std::cout << "DEBUG: Switch parsing - found nested case "
                            << currentCaseValue.value() << std::endl;
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
          std::cout << "DEBUG: Switch parsing - found default case"
                    << std::endl;
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

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertReturnStatement(const clang::ReturnStmt *returnStmt,
                                            clang::ASTContext &context) {
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
    std::cout << "Unsupported unary operator: "
              << unaryOp->getOpcodeStr(unaryOp->getOpcode()).str() << std::endl;
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
        std::cout << "Creating BufferAccessExpr for buffer: " << bufferName
                  << std::endl;
        return std::make_unique<BufferAccessExpr>(bufferName,
                                                  std::move(indexExpr));
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
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting") << " while loop"
            << std::endl;

  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;
  
  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId, mergeBlockId);
    ourStackIndex = lane.executionStack.size() - 1;
  } else {
    // Get our execution state
    auto &ourEntry = lane.executionStack[ourStackIndex];
    headerBlockId = ourEntry.loopHeaderBlockId;
    mergeBlockId = ourEntry.loopMergeBlockId;
  }

  try {
    // Get our execution state
    auto &ourEntry = lane.executionStack[ourStackIndex];
    
    // State machine for while loop execution
    switch (ourEntry.phase) {
    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      // Evaluate condition using extracted helper method
      evaluateConditionPhase(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::ExecutingBody: {
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                << " executing body for iteration " << ourEntry.loopIteration
                << " from statement " << ourEntry.statementIndex << std::endl;

      // Set up iteration-specific blocks using extracted helper method
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);

      // Execute body statements using extracted helper method
      executeBodyStatements(lane, wave, tg, ourStackIndex, headerBlockId);
      
      // Check if we need to return early (lane returned or needs resume)
      if (lane.hasReturned || lane.state != ThreadState::Ready) {
        return;
      }

      // Clean up after body execution using extracted helper method
      cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      // Handle loop exit using extracted helper method
      handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
      return;
    }

    default:
      std::cout << "ERROR: WhileStmt - Unexpected phase "
                << static_cast<int>(lane.executionStack[ourStackIndex].phase)
                << std::endl;
      return;
    }
    // }
  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - execution state is already saved
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
              << " waiting for wave operation in phase "
              << static_cast<int>(lane.executionStack[ourStackIndex].phase)
              << " at statement "
              << lane.executionStack[ourStackIndex].statementIndex
              << ", iteration "
              << lane.executionStack[ourStackIndex].loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }
  }
}

// Pure Result-based WhileStmt phase implementations
Result<bool, ExecutionError> WhileStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                                                         ThreadgroupContext &tg) {
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " evaluating condition (Result-based)" << std::endl;
  
  Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), bool, ExecutionError);
  bool shouldContinue = condVal.asBool();
  
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError> WhileStmt::executeBody(LaneContext &lane, WaveContext &wave,
                                                   ThreadgroupContext &tg) {
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing body (Result-based)" << std::endl;
  
  // Execute body statements using Result-based approach
  for (const auto &stmt : body_) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: WhileStmt - Break encountered in body" << std::endl;
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: WhileStmt - Continue encountered in body" << std::endl;
        return Ok<Unit, ExecutionError>(Unit{}); // Continue to next iteration
      } else {
        // Other errors (like WaveOperationWait) should be propagated
        return result;
      }
    }
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> WhileStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                     ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing pure Result-based while loop" << std::endl;

  // Pure Result-based while loop implementation
  while (true) {
    // Evaluate condition
    auto condResult = evaluateCondition(lane, wave, tg);
    if (condResult.is_err()) {
      return Err<Unit, ExecutionError>(condResult.unwrap_err());
    }
    
    bool shouldContinue = condResult.unwrap();
    if (!shouldContinue) {
      std::cout << "DEBUG: WhileStmt - Loop condition false, exiting" << std::endl;
      break; // Exit loop
    }
    
    // Execute body
    auto bodyResult = executeBody(lane, wave, tg);
    if (bodyResult.is_err()) {
      ExecutionError error = bodyResult.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: WhileStmt - Breaking from loop" << std::endl;
        break; // Exit loop
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: WhileStmt - Continuing loop" << std::endl;
        continue; // Continue to next iteration
      } else {
        // Other errors (like WaveOperationWait) should be propagated
        return bodyResult;
      }
    }
  }
  
  std::cout << "DEBUG: WhileStmt - Loop completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string WhileStmt::toString() const {
  std::string result = "while (" + condition_->toString() + ") {\n";
  for (const auto &stmt : body_) {
    result += "  " + stmt->toString() + "\n";
  }
  result += "}";
  return result;
}

// Helper method for fresh execution setup in WhileStmt
void WhileStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                   int ourStackIndex, uint32_t &headerBlockId, uint32_t &mergeBlockId) {
  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Get current merge stack for block creation
  std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);

  // Create loop blocks (header, merge) - pass current execution path
  auto [hBlockId, mBlockId] = tg.createLoopBlocks(static_cast<const void *>(this), parentBlockId,
                                                  currentMergeStack, lane.executionPath);
  headerBlockId = hBlockId;
  mergeBlockId = mBlockId;

  // Push merge point for loop divergence
  std::set<uint32_t> divergentBlocks = {headerBlockId};
  tg.pushMergePoint(wave.waveId, lane.laneId, static_cast<const void *>(this), parentBlockId,
                    divergentBlocks);

  // Push execution state onto stack
  LaneContext::BlockExecutionState newState(static_cast<const void *>(this),
                                            LaneContext::ControlFlowPhase::EvaluatingCondition, 0);
  newState.loopIteration = 0;
  newState.loopHeaderBlockId = headerBlockId;
  newState.loopMergeBlockId = mergeBlockId;

  lane.executionStack.push_back(newState);

  // Move to loop header block
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
}

// Helper method for condition evaluation phase in WhileStmt  
void WhileStmt::evaluateConditionPhase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                      int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration << std::endl;

  // Check loop condition
  bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return;
  }

  // Condition passed, move to body execution
  ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
  ourEntry.statementIndex = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for setting up iteration-specific blocks in WhileStmt
void WhileStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Check if we need an iteration-specific starting block
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      needsIterationBlock = true;
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                << " needs iteration block (first statement is not control flow)" << std::endl;
    } else {
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                << " no iteration block needed (first statement is control flow)" << std::endl;
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block
    iterationStartBlockId = lane.executionStack[ourStackIndex].loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x3000);

      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId, currentMergeStack, true, lane.executionPath);

      iterationStartBlockId = tg.findBlockByIdentity(iterationIdentity);

      if (iterationStartBlockId == 0) {
        std::map<WaveId, std::set<LaneId>> expectedLanes = tg.getCurrentBlockParticipants(headerBlockId);
        iterationStartBlockId = tg.findOrCreateBlockForPath(iterationIdentity, expectedLanes);
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " created iteration starting block "
                  << iterationStartBlockId << " for iteration " << ourEntry.loopIteration << std::endl;
      }

      lane.executionStack[ourStackIndex].loopBodyBlockId = iterationStartBlockId;
    }

    if (tg.getCurrentBlock(wave.waveId, lane.laneId) != iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId, wave.waveId, lane.laneId);
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " moved to iteration starting block "
                << iterationStartBlockId << std::endl;
    }
  } else {
    // No iteration block needed - but we still need to ensure unique execution context per iteration
    // Only push merge point if we're at the beginning of the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " merge stack size: " << currentMergeStack.size() << std::endl;
      for (size_t i = 0; i < currentMergeStack.size(); i++) {
        std::cout << "  Stack[" << i << "]: sourceStatement=" << currentMergeStack[i].sourceStatement << std::endl;
      }
      std::cout << "  Looking for iterationMarker=" << iterationMarker << std::endl;
      bool alreadyPushed = hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack" << std::endl;
      }

      if (!alreadyPushed) {
        std::set<uint32_t> emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId), emptyDivergentBlocks);
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " pushed iteration merge point "
                  << iterationMarker << " for iteration " << ourEntry.loopIteration << std::endl;
      }

      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing directly in current block "
                << tg.getCurrentBlock(wave.waveId, lane.laneId)
                << " (no iteration block needed, but merge stack modified)" << std::endl;
    }
  }
}

// Helper method for body statement execution in WhileStmt
void WhileStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                     int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing statement " << i
              << " in block " << blockBeforeStatement << std::endl;

    body_[i]->execute(lane, wave, tg);
    
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return" << std::endl;
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    if (lane.state != ThreadState::Ready) {
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " child statement needs resume" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement << ", Block after: " << blockAfterStatement << std::endl;
      return;
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " natural flow from block "
                << blockBeforeStatement << " to block " << blockAfterStatement
                << " during statement " << i << " (likely merge block)" << std::endl;
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
}

// Helper method for body completion cleanup in WhileStmt
void WhileStmt::cleanupAfterBodyExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                         int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Clean up iteration-specific merge point
  const void *iterationMarker = reinterpret_cast<const void *>(
      reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

  std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration " << ourEntry.loopIteration << std::endl;
  }

  // Move back to header block for next iteration
  uint32_t finalBlock = tg.membershipRegistry.getCurrentBlock(wave.waveId, lane.laneId);
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " body completed in block " << finalBlock
            << ", moving to header block " << headerBlockId
            << " for next iteration" << std::endl;
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  ourEntry.phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  ourEntry.loopIteration++;
  ourEntry.statementIndex = 0;

  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for loop exit/reconverging phase in WhileStmt
void WhileStmt::handleLoopExit(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                              int ourStackIndex, uint32_t mergeBlockId) {
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " reconverging from while loop to merge block " << mergeBlockId << std::endl;

  // Debug: Check execution stack state before popping
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " execution stack size before pop: " << lane.executionStack.size() << std::endl;

  // Pop our entry from execution stack
  lane.executionStack.pop_back();

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " execution stack size after pop: " << lane.executionStack.size() << std::endl;

  // Pop merge point and move to merge block
  tg.popMergePoint(wave.waveId, lane.laneId);
  tg.moveThreadFromUnknownToParticipating(mergeBlockId, wave.waveId, lane.laneId);

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " successfully moved to merge block " << mergeBlockId
            << " and marked as Participating" << std::endl;

  // Loop has completed, restore active state and allow progression to next statement
  lane.isActive = lane.isActive && !lane.hasReturned;

  // CRITICAL FIX: Set lane state to Ready to allow statement progression
  // This prevents infinite re-execution of the WhileStmt and allows
  // executeOneStep() to increment currentStatement
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::Ready;
  }

  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " completing reconvergence, set state to Ready for statement progression" << std::endl;
}

// Helper method for break exception handling in WhileStmt
void WhileStmt::handleBreakException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                    int ourStackIndex, uint32_t headerBlockId) {
  // Break - exit loop
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " breaking from while loop" << std::endl;
  tg.popMergePoint(wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);

  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Helper method for continue exception handling in WhileStmt
void WhileStmt::handleContinueException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                       int ourStackIndex, uint32_t headerBlockId) {
  // Continue - go to next iteration
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " continuing while loop" << std::endl;

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId, ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    std::cout << "DEBUG: WhileStmt - Marked lane " << lane.laneId << " as Left from block "
              << ourEntry.loopBodyBlockId << std::endl;
  }

  lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingCondition;
  lane.executionStack[ourStackIndex].loopIteration++;

  // Clean up - remove from all nested blocks this lane is abandoning
  if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
    tg.removeThreadFromAllSets(lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId,
                               lane.laneId);
    tg.removeThreadFromNestedBlocks(lane.executionStack[ourStackIndex].loopBodyBlockId,
                                    wave.waveId, lane.laneId);
  }

  // Move lane back to header block for proper context
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
  lane.executionStack[ourStackIndex].loopBodyBlockId = 0; // Reset for next iteration
  tg.popMergePoint(wave.waveId, lane.laneId);
  
  // Set state to WaitingForResume to prevent currentStatement increment
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Result-based versions of WhileStmt helper methods
Result<Unit, ExecutionError> WhileStmt::evaluateConditionPhase_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                      int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
            << " evaluating condition for iteration " << ourEntry.loopIteration << " (Result-based)" << std::endl;

  // Check loop condition using Result-based evaluation
  Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  bool shouldContinue = condVal.asBool();
  
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return Ok<Unit, ExecutionError>(Unit{});
  }

  // Condition passed, move to body execution
  ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingBody;
  ourEntry.statementIndex = 0;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> WhileStmt::executeBodyStatements_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                     int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements using Result-based approach
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing statement " << i
              << " in block " << blockBeforeStatement << " (Result-based)" << std::endl;

    // Use Result-based execute_result instead of exception-based execute
    TRY_RESULT(body_[i]->execute_result(lane, wave, tg), Unit, ExecutionError);
    
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return (Result-based)" << std::endl;
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " child statement needs resume (Result-based)" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement 
                << ", Block after: " << blockAfterStatement << std::endl;
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement " << i << " (likely merge block, Result-based)" << std::endl;
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

void DoWhileStmt::execute(LaneContext &lane, WaveContext &wave,
                          ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting") << " do-while loop"
            << std::endl;

  uint32_t headerBlockId = 0;
  uint32_t mergeBlockId = 0;
  
  if (!isResuming) {
    // First time execution - setup blocks and push onto execution stack
    setupFreshExecution(lane, wave, tg, ourStackIndex, headerBlockId, mergeBlockId);
    ourStackIndex = lane.executionStack.size() - 1;
  }

  // Get our execution state
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (isResuming) {
    headerBlockId = ourEntry.loopHeaderBlockId;
    mergeBlockId = ourEntry.loopMergeBlockId;
  }

  try {
    // while(lane.isActive){
    // State machine for do-while loop execution
    switch (ourEntry.phase) {
    case LaneContext::ControlFlowPhase::ExecutingBody: {
      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " executing body for iteration " << ourEntry.loopIteration
                << " from statement " << ourEntry.statementIndex << std::endl;

      // Setup iteration blocks if needed
      setupIterationBlocks(lane, wave, tg, ourStackIndex, headerBlockId);
      
      // Execute body statements
      executeBodyStatements(lane, wave, tg, ourStackIndex, headerBlockId);
      if (lane.hasReturned || lane.state != ThreadState::Ready) {
        return;
      }
      
      // Clean up after body execution
      cleanupAfterBodyExecution(lane, wave, tg, ourStackIndex, headerBlockId);
      // break;
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      evaluateConditionPhase(lane, wave, tg, ourStackIndex, headerBlockId);
      return;
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      handleLoopExit(lane, wave, tg, ourStackIndex, mergeBlockId);
      return;
    }

    default:
      std::cout << "ERROR: DoWhileStmt - Unexpected phase "
                << static_cast<int>(ourEntry.phase) << std::endl;
      return;
    }
    // }
  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - execution state is already saved
    std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
              << " waiting for wave operation in phase "
              << static_cast<int>(ourEntry.phase) << " at statement "
              << ourEntry.statementIndex << ", iteration "
              << ourEntry.loopIteration << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      handleBreakException(lane, wave, tg, ourStackIndex, headerBlockId, mergeBlockId);
      return;
    } else if (e.type == ControlFlowException::Continue) {
      handleContinueException(lane, wave, tg, ourStackIndex, headerBlockId);
      return; // Exit to prevent currentStatement increment, will resume later
    }
  }
}

// Helper methods for DoWhileStmt execute phases
void DoWhileStmt::setupFreshExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                      int ourStackIndex, uint32_t &headerBlockId, uint32_t &mergeBlockId) {
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

void DoWhileStmt::setupIterationBlocks(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                       int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Check if we need an iteration-specific starting block
  bool needsIterationBlock = false;
  uint32_t iterationStartBlockId = 0;

  // Look ahead to see if first statement (statement 0) requires unique iteration context
  if (!body_.empty()) {
    bool firstStatementIsControlFlow =
        dynamic_cast<IfStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<ForStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<WhileStmt *>(body_[0].get()) != nullptr ||
        dynamic_cast<DoWhileStmt *>(body_[0].get()) != nullptr;

    if (!firstStatementIsControlFlow) {
      needsIterationBlock = true;
      std::cout
          << "DEBUG: DoWhileStmt - Lane " << lane.laneId
          << " needs iteration block (first statement is not control flow)"
          << std::endl;
    } else {
      std::cout
          << "DEBUG: DoWhileStmt - Lane " << lane.laneId
          << " no iteration block needed (first statement is control flow)"
          << std::endl;
    }
  }

  if (needsIterationBlock) {
    // Create iteration-specific starting block
    iterationStartBlockId = ourEntry.loopBodyBlockId;
    if (iterationStartBlockId == 0) {
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);

      const void *iterationPtr = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (ourEntry.loopIteration << 16) + 0x3000);

      BlockIdentity iterationIdentity = tg.createBlockIdentity(
          iterationPtr, BlockType::REGULAR, headerBlockId,
          currentMergeStack, true, lane.executionPath);

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
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " created iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration << std::endl;
      } else {
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " found existing iteration starting block "
                  << iterationStartBlockId << " for iteration "
                  << ourEntry.loopIteration << std::endl;
      }

      ourEntry.loopBodyBlockId = iterationStartBlockId;
    }

    // Move to iteration starting block if not already there
    if (tg.getCurrentBlock(wave.waveId, lane.laneId) !=
        iterationStartBlockId) {
      tg.moveThreadFromUnknownToParticipating(iterationStartBlockId,
                                              wave.waveId, lane.laneId);
      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " moved to iteration starting block "
                << iterationStartBlockId << " for iteration "
                << ourEntry.loopIteration << std::endl;
    }
  } else {
    // No iteration block needed - but we still need to ensure unique
    // execution context per iteration Only push merge point if we're at the
    // beginning of the body (statement 0)
    if (ourEntry.statementIndex == 0) {
      // Push iteration-specific merge point so nested control flow sees
      // different merge stack
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (ourEntry.loopIteration << 16) + 0x5000);

      // Push iteration-specific merge point if not already done
      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      bool alreadyPushed =
          hasIterationMarkerInStack(currentMergeStack, iterationMarker);
      if (alreadyPushed) {
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " iteration merge point already found in merge stack"
                  << std::endl;
      }

      if (!alreadyPushed) {
        std::set<uint32_t>
            emptyDivergentBlocks; // No actual divergence, just context
        tg.pushMergePoint(wave.waveId, lane.laneId, iterationMarker,
                          tg.getCurrentBlock(wave.waveId, lane.laneId),
                          emptyDivergentBlocks);
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " pushed iteration merge point " << iterationMarker
                  << " for iteration " << ourEntry.loopIteration
                  << std::endl;
      }

      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                << " executing directly in current block "
                << tg.getCurrentBlock(wave.waveId, lane.laneId)
                << " (no iteration block needed, but merge stack modified)"
                << std::endl;
    }
  }
}

void DoWhileStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements from where we left off
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    body_[i]->execute(lane, wave, tg);
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return"
                  << std::endl;
      }

      // TODO: check if need additioanl cleanup
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }
    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return;
    }
  }
}

void DoWhileStmt::cleanupAfterBodyExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                    int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Body completed - clean up iteration-specific merge point
  const void *iterationMarker = reinterpret_cast<const void *>(
      reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) +
      0x5000);

  std::vector<MergeStackEntry> currentMergeStack =
      tg.getCurrentMergeStack(wave.waveId, lane.laneId);
  if (!currentMergeStack.empty() &&
      currentMergeStack.back().sourceStatement == iterationMarker) {
    tg.popMergePoint(wave.waveId, lane.laneId);
    std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
              << " popped iteration merge point " << iterationMarker
              << " after iteration "
              << lane.executionStack[ourStackIndex].loopIteration
              << std::endl;
  }

  // Completed body execution - move back to header block and to condition evaluation
  tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId,
                                          lane.laneId);
  ourEntry.loopBodyBlockId = 0; // Reset for next iteration
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " completed body for iteration "
            << lane.executionStack[ourStackIndex].loopIteration
            << std::endl;
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::EvaluatingCondition;
}

void DoWhileStmt::evaluateConditionPhase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                 int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " evaluating condition after iteration "
            << ourEntry.loopIteration << std::endl;

  // Check loop condition
  bool shouldContinue = condition_->evaluate(lane, wave, tg).asBool();
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
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return; // Exit to prevent currentStatement increment, will resume later
  }

  // Condition passed, move to next iteration body execution
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].loopIteration++;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  // break;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  return; // Exit to prevent currentStatement increment, will resume later
}

void DoWhileStmt::handleLoopExit(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                         int ourStackIndex, uint32_t mergeBlockId) {
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " reconverging from do-while loop" << std::endl;

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

void DoWhileStmt::handleBreakException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                               int ourStackIndex, uint32_t headerBlockId, uint32_t mergeBlockId) {
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " breaking from do-while loop" << std::endl;
  tg.popMergePoint(wave.waveId, lane.laneId);

  // Clean up - remove from blocks this lane will never reach
  tg.removeThreadFromAllSets(headerBlockId, wave.waveId, lane.laneId);
  tg.removeThreadFromNestedBlocks(headerBlockId, wave.waveId, lane.laneId);

  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

void DoWhileStmt::handleContinueException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                  int ourStackIndex, uint32_t headerBlockId) {
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " continuing do-while loop" << std::endl;

  // CRITICAL FIX: Mark lane as Left from current block when continuing
  auto &ourEntry = lane.executionStack[ourStackIndex];
  if (ourEntry.loopBodyBlockId != 0) {
    tg.membershipRegistry.setLaneStatus(wave.waveId, lane.laneId,
                                        ourEntry.loopBodyBlockId,
                                        LaneBlockStatus::Left);
    std::cout << "DEBUG: DoWhileStmt - Marked lane " << lane.laneId
              << " as Left from block " << ourEntry.loopBodyBlockId
              << std::endl;
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
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
}

// Phase-based Result methods for DoWhileStmt
Result<Unit, ExecutionError> DoWhileStmt::executeBody(LaneContext &lane, WaveContext &wave,
                                                    ThreadgroupContext &tg) {
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " executing body (Result-based)" << std::endl;
  
  for (const auto &stmt : body_) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      std::cout << "DEBUG: DoWhileStmt - Body encountered error" << std::endl;
      
      // Break and continue are normal control flow for do-while loops
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: DoWhileStmt - Break encountered in body" << std::endl;
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: DoWhileStmt - Continue encountered in body" << std::endl;
        return Err<Unit, ExecutionError>(ExecutionError::ControlFlowContinue);
      }
      
      // Other errors propagate up
      return result;
    }
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<bool, ExecutionError> DoWhileStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                                                          ThreadgroupContext &tg) {
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " evaluating condition (Result-based)" << std::endl;
  
  auto condResult = condition_->evaluate_result(lane, wave, tg);
  if (condResult.is_err()) {
    return Err<bool, ExecutionError>(condResult.unwrap_err());
  }
  
  Value condVal = condResult.unwrap();
  bool shouldContinue = condVal.asBool();
  
  std::cout << "DEBUG: DoWhileStmt - Condition result: " << shouldContinue << std::endl;
  return Ok<bool, ExecutionError>(shouldContinue);
}

Result<Unit, ExecutionError> DoWhileStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " executing pure Result-based do-while loop" << std::endl;
  
  // do-while loop: execute body first, then check condition
  do {
    // Execute body phase
    auto bodyResult = executeBody(lane, wave, tg);
    if (bodyResult.is_err()) {
      ExecutionError error = bodyResult.unwrap_err();
      
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: DoWhileStmt - Breaking from loop" << std::endl;
        break;
      } else if (error == ExecutionError::ControlFlowContinue) {
        std::cout << "DEBUG: DoWhileStmt - Continuing to condition check" << std::endl;
        // Continue to condition evaluation
      } else {
        // Other errors propagate up
        return bodyResult;
      }
    }
    
    // Evaluate condition phase
    auto condResult = evaluateCondition(lane, wave, tg);
    if (condResult.is_err()) {
      return Err<Unit, ExecutionError>(condResult.unwrap_err());
    }
    
    bool shouldContinue = condResult.unwrap();
    if (!shouldContinue) {
      break;
    }
    
  } while (true);
  
  std::cout << "DEBUG: DoWhileStmt - Loop completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

std::string DoWhileStmt::toString() const {
  std::string result = "do {\n";
  for (const auto &stmt : body_) {
    result += "  " + stmt->toString() + "\n";
  }
  result += "} while (" + condition_->toString() + ");";
  return result;
}

// Result-based versions of DoWhileStmt helper methods
Result<Unit, ExecutionError> DoWhileStmt::executeBodyStatements_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements from where we left off using Result-based approach
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    
    // Use Result-based execute_result instead of exception-based execute
    TRY_RESULT(body_[i]->execute_result(lane, wave, tg), Unit, ExecutionError);
    
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return (Result-based)"
                  << std::endl;
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return Ok<Unit, ExecutionError>(Unit{});
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId << " child statement needs resume (Result-based)" << std::endl;
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Update statement index  
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> DoWhileStmt::evaluateConditionPhase_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                 int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
            << " evaluating condition after iteration "
            << ourEntry.loopIteration << " (Result-based)" << std::endl;

  // Check loop condition using Result-based evaluation
  Value condVal = TRY_RESULT(condition_->evaluate_result(lane, wave, tg), Unit, ExecutionError);
  bool shouldContinue = condVal.asBool();
  
  if (!shouldContinue) {
    // Lane is exiting loop - comprehensive cleanup from header and all iteration blocks
    tg.removeThreadFromAllSets(headerBlockId, wave.waveId,
                               lane.laneId); // Remove from header
    tg.removeThreadFromNestedBlocks(
        headerBlockId, wave.waveId,
        lane.laneId); // Remove from iteration blocks

    // Move to reconverging phase
    lane.executionStack[ourStackIndex].phase =
        LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return Ok<Unit, ExecutionError>(Unit{}); // Exit to prevent currentStatement increment, will resume later
  }

  // Condition passed, move to next iteration body execution
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::ExecutingBody;
  lane.executionStack[ourStackIndex].statementIndex = 0;
  lane.executionStack[ourStackIndex].loopIteration++;
  if (!isProtectedState(lane.state)) {
    lane.state = ThreadState::WaitingForResume;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

// SwitchStmt implementation
SwitchStmt::SwitchStmt(std::unique_ptr<Expression> cond)
    : condition_(std::move(cond)) {}

void SwitchStmt::addCase(int value,
                         std::vector<std::unique_ptr<Statement>> stmts) {
  std::cout << "DEBUG: SwitchStmt::addCase - adding case " << value
            << std::endl;
  cases_.push_back({value, std::move(stmts)});
}

void SwitchStmt::addDefault(std::vector<std::unique_ptr<Statement>> stmts) {
  std::cout << "DEBUG: SwitchStmt::addDefault - adding default case"
            << std::endl;
  cases_.push_back({std::nullopt, std::move(stmts)});
}

void SwitchStmt::execute(LaneContext &lane, WaveContext &wave,
                         ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " "
            << (isResuming ? "resuming" : "starting") << " switch statement"
            << std::endl;

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " starting fresh execution (pushed to stack depth="
              << lane.executionStack.size() << ", this=" << this << ")"
              << std::endl;

    // Setup switch execution blocks
    setupSwitchExecution(lane, wave, tg, ourStackIndex);
  } else {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " resuming execution (found at stack index=" << ourStackIndex
              << ", current stack depth=" << lane.executionStack.size()
              << ", this=" << this << ")" << std::endl;
  }

  try {
    auto &ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " in phase "
              << LaneContext::getPhaseString(ourEntry.phase)
              << " (stack depth=" << lane.executionStack.size()
              << ", our index=" << ourStackIndex << ", this=" << this << ")"
              << std::endl;

    switch (ourEntry.phase) {
    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      // Evaluate switch value
      evaluateSwitchValue(lane, wave, tg, ourStackIndex);
      
      // Find matching case and set up execution
      findMatchingCase(lane, wave, tg, ourStackIndex);
      
      // Check if we're moving to reconverging (no matching case)
      if (ourEntry.phase == LaneContext::ControlFlowPhase::Reconverging) {
        return;
      }
      
      // Set state to WaitingForResume to prevent currentStatement increment
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return; // Exit to prevent currentStatement increment, will resume later
    }

    case LaneContext::ControlFlowPhase::ExecutingCase: {
      executeCaseStatements(lane, wave, tg, ourStackIndex);
      if (lane.hasReturned || lane.state != ThreadState::Ready) {
        return;
      }
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return;
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      handleReconvergence(lane, wave, tg, ourStackIndex);
      return;
    }

    default:
      std::cout << "ERROR: SwitchStmt - Lane " << lane.laneId
                << " unexpected phase" << std::endl;
      // TODO: verify if need additional step
      lane.executionStack.pop_back();
      return;
    }

  } catch (const WaveOperationWaitException &) {
    // Wave operation is waiting - execution state is already saved
    auto &ourEntry = lane.executionStack[ourStackIndex];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " waiting for wave operation in case " << ourEntry.caseIndex
              << " at statement " << ourEntry.statementIndex << std::endl;
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    if (e.type == ControlFlowException::Break) {
      handleBreakException(lane, wave, tg, ourStackIndex);
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return;
    }
    // Continue statements don't apply to switch - just ignore them
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " ignoring continue in switch" << std::endl;
  }
}

// Helper methods for SwitchStmt execute phases
void SwitchStmt::setupSwitchExecution(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
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
    std::cout << "ERROR: SwitchStmt - Insufficient blocks created"
              << std::endl;
    return;
  }

  ourEntry.switchHeaderBlockId =
      allBlockIds[0]; // First element is header block
  ourEntry.switchMergeBlockId =
      allBlockIds.back(); // Last element is merge block

  // Extract case blocks (everything between header and merge)
  ourEntry.switchCaseBlockIds =
      std::vector<uint32_t>(allBlockIds.begin() + 1, allBlockIds.end() - 1);

  std::cout << "DEBUG: SwitchStmt - Created header block "
            << ourEntry.switchHeaderBlockId << ", "
            << ourEntry.switchCaseBlockIds.size()
            << " case blocks, and merge block " << ourEntry.switchMergeBlockId
            << std::endl;

  // Move to header block for condition evaluation
  tg.moveThreadFromUnknownToParticipating(ourEntry.switchHeaderBlockId,
                                          wave.waveId, lane.laneId);
}

void SwitchStmt::evaluateSwitchValue(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                         int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " evaluating switch condition" << std::endl;

  // Only evaluate condition if not already evaluated
  if (!ourEntry.conditionEvaluated) {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " evaluating condition for first time" << std::endl;
    auto condValue = condition_->evaluate(lane, wave, tg);
    lane.executionStack[ourStackIndex].switchValue = condValue;
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " switch condition evaluated to: " << condValue.asInt()
              << std::endl;
  } else {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " using cached condition result="
              << ourEntry.switchValue.asInt() << std::endl;
  }
}

void SwitchStmt::findMatchingCase(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                      int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Find which case this lane should execute
  int switchValue = ourEntry.switchValue.asInt();
  size_t matchingCaseIndex = SIZE_MAX;

  for (size_t i = 0; i < cases_.size(); ++i) {
    if (cases_[i].value.has_value() &&
        cases_[i].value.value() == switchValue) {
      matchingCaseIndex = i;
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " matched case " << cases_[i].value.value()
                << " at index " << i << std::endl;
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
    // No matching case - remove from all case blocks and exit switch via reconvergence
    std::cout
        << "DEBUG: SwitchStmt - Lane " << lane.laneId
        << " no matching case found, cleaning up and entering reconvergence"
        << std::endl;

    // Remove this lane from ALL case blocks (it will never execute any cases)
    for (size_t i = 0; i < ourEntry.switchCaseBlockIds.size(); ++i) {
      uint32_t caseBlockId = ourEntry.switchCaseBlockIds[i];
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " removing from case block " << caseBlockId << " (case "
                << i << ") - no matching case" << std::endl;
      tg.removeThreadFromAllSets(caseBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(caseBlockId, wave.waveId,
                                      lane.laneId);
    }

    ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
    if (!isProtectedState(lane.state)) {
      lane.state = ThreadState::WaitingForResume;
    }
    return;
  }

  // Set up for case execution and move to appropriate block
  ourEntry.caseIndex = matchingCaseIndex;
  ourEntry.statementIndex = 0;
  ourEntry.phase = LaneContext::ControlFlowPhase::ExecutingCase;

  // Move lane to the appropriate case block and remove from previous cases
  if (matchingCaseIndex < ourEntry.switchCaseBlockIds.size()) {
    uint32_t chosenCaseBlockId =
        ourEntry.switchCaseBlockIds[matchingCaseIndex];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " moving to case block " << chosenCaseBlockId
              << " for case " << matchingCaseIndex << std::endl;

    // Move to the first matching case block only (like if/loop pattern)
    uint32_t firstCaseBlockId =
        ourEntry.switchCaseBlockIds[matchingCaseIndex];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " moving from header to first case block "
              << firstCaseBlockId << " (case " << matchingCaseIndex << ")"
              << std::endl;

    // tg.assignLaneToBlock(wave.waveId, lane.laneId, firstCaseBlockId);
    tg.moveThreadFromUnknownToParticipating(firstCaseBlockId, wave.waveId,
                                            lane.laneId);

    // Remove from all previous case blocks (cases this lane will never execute)
    for (size_t i = 0; i < matchingCaseIndex; ++i) {
      uint32_t previousCaseBlockId = ourEntry.switchCaseBlockIds[i];
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " removing from previous case block "
                << previousCaseBlockId << " (case " << i << ")"
                << std::endl;
      tg.removeThreadFromUnknown(previousCaseBlockId, wave.waveId,
                                 lane.laneId);
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

void SwitchStmt::executeCaseStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                           int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute all statements from current position until case/switch completion
  while (ourEntry.caseIndex < cases_.size()) {
    const auto &caseBlock = cases_[ourEntry.caseIndex];
    std::string caseLabel = caseBlock.value.has_value()
                                ? std::to_string(caseBlock.value.value())
                                : "default";

    // Execute all statements in current case from saved position
    for (size_t i = ourEntry.statementIndex;
         i < caseBlock.statements.size(); i++) {
      lane.executionStack[ourStackIndex].statementIndex = i;

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " executing statement " << i << " in case " << caseLabel
                << std::endl;

      // Execute the current statement
      caseBlock.statements[i]->execute(lane, wave, tg);

      if (lane.hasReturned) {
        std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " popping stack due to return (depth "
                  << lane.executionStack.size() << "->"
                  << (lane.executionStack.size() - 1) << ", this=" << this
                  << ")" << std::endl;
        // TODO: verify if need additional cleanup
        lane.executionStack.pop_back();
        tg.popMergePoint(wave.waveId, lane.laneId);
        return;
      }

      if (lane.state != ThreadState::Ready) {
        // Child statement needs to resume - don't continue
        return;
      }
      lane.executionStack[ourStackIndex].statementIndex = i + 1;
    }

    // Check if lane is ready before fallthrough - if waiting for wave
    // operations, don't move to next case
    if (lane.state != ThreadState::Ready) {
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " completed case " << caseLabel
                << " but is not Ready (state=" << (int)lane.state
                << "), pausing before fallthrough" << std::endl;
      return;
    }

    // Current case completed - move to next case (fallthrough)
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " completed case " << caseLabel
              << ", falling through to next" << std::endl;

    size_t nextCaseIndex = ourEntry.caseIndex + 1;

    // Move lane to next case block if it exists
    if (nextCaseIndex < ourEntry.switchCaseBlockIds.size()) {
      uint32_t currentCaseBlockId =
          ourEntry.switchCaseBlockIds[ourEntry.caseIndex];
      uint32_t nextCaseBlockId = ourEntry.switchCaseBlockIds[nextCaseIndex];

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " moving from case block " << currentCaseBlockId
                << " to case block " << nextCaseBlockId << " (fallthrough)"
                << std::endl;

      // Move to next case block (fallthrough)
      // tg.assignLaneToBlock(wave.waveId, lane.laneId, nextCaseBlockId);
      tg.moveThreadFromUnknownToParticipating(nextCaseBlockId, wave.waveId,
                                              lane.laneId);

      // Remove from current case block
      tg.removeThreadFromAllSets(currentCaseBlockId, wave.waveId,
                                 lane.laneId);
    }

    // Move to next case
    lane.executionStack[ourStackIndex].caseIndex++;
    lane.executionStack[ourStackIndex].statementIndex = 0;
  }

  // All cases completed - enter reconvergence
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " completed all cases, entering reconvergence" << std::endl;
  ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
}

void SwitchStmt::handleReconvergence(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                         int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " reconverging from switch" << std::endl;

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

void SwitchStmt::handleBreakException(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                          int ourStackIndex) {
  // Break - exit switch and clean up subsequent case blocks
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " breaking from switch" << std::endl;

  auto &ourEntry = lane.executionStack[ourStackIndex];

  // Remove this lane from ALL case blocks (it's breaking out of the entire switch)
  for (size_t i = 0; i < ourEntry.switchCaseBlockIds.size(); ++i) {
    uint32_t caseBlockId = ourEntry.switchCaseBlockIds[i];
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " removing from case block " << caseBlockId << " (case "
              << i << ") due to break" << std::endl;
    tg.removeThreadFromAllSets(caseBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(caseBlockId, wave.waveId, lane.laneId);
  }
  // Use Reconverging phase instead of direct exit
  lane.executionStack[ourStackIndex].phase =
      LaneContext::ControlFlowPhase::Reconverging;
}

// Phase-based Result methods for SwitchStmt
Result<int, ExecutionError> SwitchStmt::evaluateCondition(LaneContext &lane, WaveContext &wave,
                                                        ThreadgroupContext &tg) {
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " evaluating condition (Result-based)" << std::endl;
  
  auto condResult = condition_->evaluate_result(lane, wave, tg);
  if (condResult.is_err()) {
    return Err<int, ExecutionError>(condResult.unwrap_err());
  }
  
  Value condVal = condResult.unwrap();
  int switchValue = condVal.asInt();
  
  std::cout << "DEBUG: SwitchStmt - Switch value: " << switchValue << std::endl;
  return Ok<int, ExecutionError>(switchValue);
}

Result<Unit, ExecutionError> SwitchStmt::executeCase(size_t caseIndex, LaneContext &lane, 
                                                    WaveContext &wave, ThreadgroupContext &tg) {
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " executing case " << caseIndex << " (Result-based)" << std::endl;
  
  const auto &caseBlock = cases_[caseIndex];
  
  for (const auto &stmt : caseBlock.statements) {
    auto result = stmt->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      std::cout << "DEBUG: SwitchStmt - Case encountered error" << std::endl;
      
      // Break exits the switch (normal behavior)
      if (error == ExecutionError::ControlFlowBreak) {
        std::cout << "DEBUG: SwitchStmt - Break encountered in case" << std::endl;
        return Ok<Unit, ExecutionError>(Unit{}); // Break stops switch execution successfully
      }
      
      // Continue should be handled by containing loop, not switch
      // Other errors propagate up
      return result;
    }
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> SwitchStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                      ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});

  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId << " executing pure Result-based switch statement" << std::endl;
  
  // Evaluate condition phase
  auto condResult = evaluateCondition(lane, wave, tg);
  if (condResult.is_err()) {
    return Err<Unit, ExecutionError>(condResult.unwrap_err());
  }
  
  int switchValue = condResult.unwrap();
  
  // Find matching case
  bool foundMatch = false;
  bool executingCases = false;
  
  for (size_t i = 0; i < cases_.size(); ++i) {
    const auto &caseBlock = cases_[i];
    
    // Check if this case matches or if we're in fall-through mode
    if (!executingCases) {
      if (caseBlock.value.has_value() && caseBlock.value.value() == switchValue) {
        std::cout << "DEBUG: SwitchStmt - Found matching case: " << switchValue << std::endl;
        foundMatch = true;
        executingCases = true;
      } else if (!caseBlock.value.has_value()) {
        // Default case
        std::cout << "DEBUG: SwitchStmt - Executing default case" << std::endl;
        foundMatch = true;
        executingCases = true;
      }
    }
    
    // Execute case if we're in execution mode
    if (executingCases) {
      auto caseResult = executeCase(i, lane, wave, tg);
      if (caseResult.is_err()) {
        return caseResult; // Propagate errors
      }
      
      // Note: This implementation doesn't handle explicit break detection for fall-through
      // If no break is encountered, we continue to the next case (fall-through behavior)
      // In a full implementation, we'd need to track if a break was executed
    }
  }
  
  if (!foundMatch) {
    std::cout << "DEBUG: SwitchStmt - No matching case found, switch completes" << std::endl;
  }
  
  std::cout << "DEBUG: SwitchStmt - Switch completed successfully" << std::endl;
  return Ok<Unit, ExecutionError>(Unit{});
}

// Result-based helper functions for SwitchStmt
Result<Unit, ExecutionError> SwitchStmt::evaluateSwitchValue_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                         int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " evaluating switch condition (Result-based)" << std::endl;

  // Only evaluate condition if not already evaluated
  if (!ourEntry.conditionEvaluated) {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " evaluating condition for first time" << std::endl;
    
    auto condValue = condition_->evaluate_result(lane, wave, tg);
    if (condValue.is_err()) {
      return Err<Unit, ExecutionError>(condValue.unwrap_err());
    }
    
    lane.executionStack[ourStackIndex].switchValue = condValue.unwrap();
    lane.executionStack[ourStackIndex].conditionEvaluated = true;
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " switch condition evaluated to: " << condValue.unwrap().asInt()
              << std::endl;
  } else {
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " using cached condition result="
              << ourEntry.switchValue.asInt() << std::endl;
  }
  
  return Ok<Unit, ExecutionError>(Unit{});
}

Result<Unit, ExecutionError> SwitchStmt::executeCaseStatements_result(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                           int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute all statements from current position until case/switch completion
  while (ourEntry.caseIndex < cases_.size()) {
    const auto &caseBlock = cases_[ourEntry.caseIndex];
    std::string caseLabel = caseBlock.value.has_value()
                                ? std::to_string(caseBlock.value.value())
                                : "default";

    // Execute all statements in current case from saved position
    for (size_t i = ourEntry.statementIndex;
         i < caseBlock.statements.size(); i++) {
      lane.executionStack[ourStackIndex].statementIndex = i;

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " executing statement " << i << " in case " << caseLabel
                << " (Result-based)" << std::endl;

      // Use Result-based execute_result instead of exception-based execute
      TRY_RESULT(caseBlock.statements[i]->execute_result(lane, wave, tg), Unit, ExecutionError);

      if (lane.hasReturned) {
        std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " popping stack due to return (depth "
                  << lane.executionStack.size() << "->"
                  << (lane.executionStack.size() - 1) << ", this=" << this
                  << ")" << std::endl;
        // Clean up and return
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

    // Check if lane is ready before fallthrough - if waiting for wave
    // operations, don't move to next case
    if (lane.state != ThreadState::Ready) {
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " completed case " << caseLabel
                << " but is not Ready (state=" << (int)lane.state
                << "), pausing before fallthrough" << std::endl;
      return Ok<Unit, ExecutionError>(Unit{});
    }

    // Current case completed - move to next case (fallthrough)
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " completed case " << caseLabel
              << ", falling through to next" << std::endl;

    size_t nextCaseIndex = ourEntry.caseIndex + 1;

    // Move lane to next case block if it exists
    if (nextCaseIndex < ourEntry.switchCaseBlockIds.size()) {
      uint32_t currentCaseBlockId =
          ourEntry.switchCaseBlockIds[ourEntry.caseIndex];
      uint32_t nextCaseBlockId = ourEntry.switchCaseBlockIds[nextCaseIndex];

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " moving from case block " << currentCaseBlockId
                << " to case block " << nextCaseBlockId << " (fallthrough)"
                << std::endl;

      // Move to next case block (fallthrough)
      tg.moveThreadFromUnknownToParticipating(nextCaseBlockId, wave.waveId,
                                              lane.laneId);

      // Remove from current case block
      tg.removeThreadFromAllSets(currentCaseBlockId, wave.waveId,
                                 lane.laneId);
    }

    // Move to next case
    lane.executionStack[ourStackIndex].caseIndex++;
    lane.executionStack[ourStackIndex].statementIndex = 0;
  }

  // All cases completed - enter reconvergence
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " completed all cases, entering reconvergence" << std::endl;
  ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
  
  return Ok<Unit, ExecutionError>(Unit{});
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

Result<Unit, ExecutionError> BreakStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                     ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});
  
  // Pure Result-based break - no exceptions thrown!
  std::cout << "DEBUG: BreakStmt - Lane " << lane.laneId
            << " executing break via Result" << std::endl;
  return Err<Unit, ExecutionError>(ExecutionError::ControlFlowBreak);
}

// ContinueStmt implementation
void ContinueStmt::execute(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;
  throw ControlFlowException(ControlFlowException::Continue);
}

Result<Unit, ExecutionError> ContinueStmt::execute_result(LaneContext &lane, WaveContext &wave,
                                                        ThreadgroupContext &tg) {
  if (!lane.isActive)
    return Ok<Unit, ExecutionError>(Unit{});
  
  // Pure Result-based continue - no exceptions thrown!
  std::cout << "DEBUG: ContinueStmt - Lane " << lane.laneId
            << " executing continue via Result" << std::endl;
  return Err<Unit, ExecutionError>(ExecutionError::ControlFlowContinue);
}

// Dynamic execution block methods
uint32_t ThreadgroupContext::createExecutionBlock(
    const std::map<WaveId, std::set<LaneId>> &lanes, const void *sourceStmt) {
  uint32_t blockId = nextBlockId++;

  DynamicExecutionBlock block;
  block.setBlockId(blockId);
  for (const auto &[waveId, laneSet] : lanes) {
    for (LaneId laneId : laneSet) {
      // Registry is the single source of truth
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::Participating);
    }
  }
  block.setProgramPoint(0);
  block.setSourceStatement(sourceStmt);

  // Calculate total lanes across all waves
  size_t totalLanes = 0;
  for (const auto &[waveId, laneSet] : lanes) {
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
  std::cout << "DEBUG: assignLaneToBlock - START: lane " << laneId
            << " being assigned to block " << blockId << std::endl;

  // Show current state before assignment
  auto partBefore = membershipRegistry.getParticipatingLanes(waveId, blockId);
  auto waitBefore = membershipRegistry.getWaitingLanes(waveId, blockId);
  auto unknownBefore = membershipRegistry.getUnknownLanes(waveId, blockId);
  std::cout << "DEBUG: assignLaneToBlock - BEFORE: block " << blockId << " has "
            << partBefore.size() << " participating lanes";
  if (!partBefore.empty()) {
    std::cout << " (";
    for (auto it = partBefore.begin(); it != partBefore.end(); ++it) {
      if (it != partBefore.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << ", " << waitBefore.size() << " waiting lanes";
  if (!waitBefore.empty()) {
    std::cout << " (";
    for (auto it = waitBefore.begin(); it != waitBefore.end(); ++it) {
      if (it != waitBefore.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << ", " << unknownBefore.size() << " unknown lanes";
  if (!unknownBefore.empty()) {
    std::cout << " (";
    for (auto it = unknownBefore.begin(); it != unknownBefore.end(); ++it) {
      if (it != unknownBefore.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << std::endl;

  // Don't move lanes that are waiting for wave operations or barriers
  if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
    auto state = waves[waveId]->lanes[laneId]->state;
    if (state == ThreadState::WaitingForWave ||
        state == ThreadState::WaitingAtBarrier) {
      std::cout << "DEBUG: Preventing move of waiting lane " << laneId
                << " (state=" << (int)state << ") to block " << blockId
                << std::endl;
      return;
    }
  }

  // Check if lane is already in the target block
  uint32_t currentBlockId = membershipRegistry.getCurrentBlock(waveId, laneId);
  if (currentBlockId == blockId) {
    std::cout << "DEBUG: assignLaneToBlock - lane " << laneId
              << " already in block " << blockId
              << ", skipping redundant assignment" << std::endl;
    return;
  }

  // Remove lane from its current block if it's in one
  if (currentBlockId != 0) {
    auto* oldBlock = getBlock(currentBlockId);
    if (oldBlock) {
      // NEW: Don't remove from arrivedLanes if moving from header to iteration
      // block This keeps lanes in header's arrivedLanes until they exit the
      // loop entirely
      const auto* newBlock = getBlock(blockId);
      bool isHeaderToLoopBody =
          (oldBlock->getBlockType() == BlockType::LOOP_HEADER &&
           newBlock &&
           newBlock->getBlockType() != BlockType::LOOP_EXIT);

      std::cout << "DEBUG: assignLaneToBlock - moving lane " << laneId
                << " from block " << currentBlockId << " (type "
                << (int)oldBlock->getBlockType() << ") to block "
                << blockId << " (type "
                << (int)(newBlock ? newBlock->getBlockType() : BlockType::REGULAR)
                << "), isHeaderToLoopBody=" << isHeaderToLoopBody << std::endl;

      if (!isHeaderToLoopBody) {
        // Remove lane from old block
        membershipRegistry.setLaneStatus(waveId, laneId, currentBlockId,
                                         LaneBlockStatus::Left);
        std::cout << "DEBUG: Removed lane " << laneId << " from block "
                  << currentBlockId << std::endl;
      } else {
        // Keep lane as Participating in header block for proper unknown lane
        // tracking This allows nested blocks to correctly determine expected
        // participants
        std::cout << "DEBUG: Keeping lane " << laneId
                  << " as Participating in header block " << currentBlockId
                  << " while also adding to loop body block " << blockId
                  << std::endl;
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
  std::cout << "DEBUG: assignLaneToBlock - AFTER: block " << blockId << " has "
            << partAfter.size() << " participating lanes";
  if (!partAfter.empty()) {
    std::cout << " (";
    for (auto it = partAfter.begin(); it != partAfter.end(); ++it) {
      if (it != partAfter.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << ", " << waitAfter.size() << " waiting lanes";
  if (!waitAfter.empty()) {
    std::cout << " (";
    for (auto it = waitAfter.begin(); it != waitAfter.end(); ++it) {
      if (it != waitAfter.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << ", " << unknownAfter.size() << " unknown lanes";
  if (!unknownAfter.empty()) {
    std::cout << " (";
    for (auto it = unknownAfter.begin(); it != unknownAfter.end(); ++it) {
      if (it != unknownAfter.begin())
        std::cout << " ";
      std::cout << *it;
    }
    std::cout << ")";
  }
  std::cout << std::endl;

  std::cout << "DEBUG: assignLaneToBlock - END: lane " << laneId
            << " successfully assigned to block " << blockId << std::endl;

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
    std::cout << "WARNING: getCurrentBlock - Lane " << laneId
              << " found in multiple blocks: ";
    for (size_t i = 0; i < foundInBlocks.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << foundInBlocks[i];
    }
    std::cout << " (registry returned: " << resultBlockId << ")" << std::endl;
  }

  return resultBlockId;
}

void ThreadgroupContext::mergeExecutionPaths(
    const std::vector<uint32_t> &blockIds, uint32_t targetBlockId) {
  // Create or update the target block
  std::map<WaveId, std::set<LaneId>> mergedLanes;

  for (uint32_t blockId : blockIds) {
    const auto* block = getBlock(blockId);
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
  const auto* block = getBlock(blockId);
  if (block) {
    // Update BlockMembershipRegistry: lane arrived and is no longer unknown
    // First check current status to handle the transition properly
    LaneBlockStatus currentStatus =
        membershipRegistry.getLaneStatus(waveId, laneId, blockId);

    // If lane was unknown, it's now participating (arrived)
    if (currentStatus == LaneBlockStatus::Unknown) {
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::Participating);
      std::cout << "DEBUG: markLaneArrived - Lane " << laneId
                << " transitioned from Unknown to Participating in block "
                << blockId << std::endl;
    } else {
      // Lane was already participating, just ensure it stays participating
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::Participating);
      std::cout << "DEBUG: markLaneArrived - Lane " << laneId
                << " confirmed as Participating in block " << blockId
                << std::endl;
    }

    // Validate with registry
    bool registryResolved =
        membershipRegistry.isWaveAllUnknownResolved(waveId, blockId);
    bool oldResolved =
        membershipRegistry.getUnknownLanes(waveId, blockId).empty();

    if (registryResolved != oldResolved) {
      std::cout << "INFO: markLaneArrived - Block " << blockId << " wave "
                << waveId
                << " resolution difference - registry: " << registryResolved
                << ", old: " << oldResolved << " (tracked by registry)"
                << std::endl;
    }

    // Resolution status tracked by registry - no need for old system metadata
  }
}

void ThreadgroupContext::markLaneWaitingForWave(WaveId waveId, LaneId laneId,
                                                uint32_t blockId) {
  std::cout << "DEBUG: markLaneWaitingForWave - Lane " << laneId << " wave "
            << waveId << " in block " << blockId << std::endl;

  const auto* block = getBlock(blockId);
  if (block) {
    // it->second.addWaitingLane(waveId, laneId);
    // Change lane state to waiting
    if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
      waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;
      // Also update the BlockMembershipRegistry
      membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                       LaneBlockStatus::WaitingForWave);
      std::cout << "DEBUG: markLaneWaitingForWave - Successfully set lane "
                << laneId << " to WaitingForWave in block " << blockId
                << std::endl;
    }
  } else {
    std::cout << "DEBUG: markLaneWaitingForWave - Block " << blockId
              << " not found!" << std::endl;
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
  const auto* block = getBlock(blockId);
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
  const auto* block = getBlock(blockId);
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

  std::cout << "DEBUG: findOrCreateBlockForPath called with "
            << unknownLanes.size() << " waves of unknown lanes" << std::endl;
  for (const auto &[waveId, laneSet] : unknownLanes) {
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
    // The existing block already has the correct unknown lanes from when it was
    // first created. Those lanes will be properly removed by
    // removeThreadFromNestedBlocks when appropriate.
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
  for (const auto &[waveId, laneSet] : unknownLanes) {
    for (LaneId laneId : laneSet) {
      std::cout << "DEBUG: addUnknownLane - adding lane " << laneId
                << " to new block " << newBlockId << std::endl;
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
  instrIdentity.instructionType =
      "WaveActiveOp"; // This should be passed as parameter in a full
                      // implementation
  instrIdentity.sourceExpression = instruction;

  // Use the wave-specific approach for instruction synchronization
  bool canExecuteInBlock =
      canExecuteWaveInstructionInBlock(currentBlockId, waveId, instrIdentity);

  // Debug unknown lanes state
  const auto* block = getBlock(currentBlockId);
  if (block) {
    auto unknownLanes =
        membershipRegistry.getUnknownLanes(waveId, currentBlockId);
    std::cout << "DEBUG: Block " << currentBlockId << " wave " << waveId
              << " unknown lanes: {";
    for (LaneId uid : unknownLanes) {
      std::cout << uid << " ";
    }
    std::cout << "}" << std::endl;
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

  std::cout << "DEBUG: canExecuteWaveInstruction for lane " << laneId
            << " in block " << currentBlockId
            << ": canExecuteInBlock=" << canExecuteInBlock
            << ", allParticipantsKnown=" << allParticipantsKnown
            << ", allParticipantsArrived=" << allParticipantsArrived
            << ", canExecuteGlobal=" << canExecuteGlobal
            << ", syncPointPhase=" << phaseStr << std::endl;

  // Both approaches should agree for proper synchronization
  return canExecuteInBlock && canExecuteGlobal;
}

void ThreadgroupContext::markLaneWaitingAtWaveInstruction(
    WaveId waveId, LaneId laneId, const void *instruction,
    const std::string &instructionType) {
  // Create instruction identity
  InstructionIdentity instrIdentity =
      createInstructionIdentity(instruction, instructionType, instruction);

  // Get current block for this lane
  uint32_t currentBlockId = getCurrentBlock(waveId, laneId);
  if (currentBlockId == 0) {
    std::cout << "ERROR: Lane " << laneId << " in wave " << waveId
              << " not assigned to any block when arriving at instruction "
              << instructionType << std::endl;
    // Debug: Check registry state
    std::cout << "DEBUG: Checking all blocks for lane " << laneId << "..."
              << std::endl;
    for (const auto &[blockId, block] : executionBlocks) {
      auto status = membershipRegistry.getLaneStatus(waveId, laneId, blockId);
      if (status != LaneBlockStatus::Unknown &&
          status != LaneBlockStatus::Left) {
        std::cout << "  Found lane in block " << blockId << " with status "
                  << (int)status << std::endl;
      }
    }
    throw std::runtime_error(
        "Lane not assigned to any block when arriving at instruction");
  }

  // Add instruction to block with this lane as participant
  std::map<WaveId, std::set<LaneId>> participants = {{waveId, {laneId}}};
  addInstructionToBlock(currentBlockId, instrIdentity, participants);

  // Create or update sync point for this instruction
  createOrUpdateWaveSyncPoint(instruction, waveId, laneId, instructionType);

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
    const auto* block = getBlock(blockId);
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
    std::cout << "WARNING: Consistency mismatch - syncPoint says "
              << syncPointKnown << " but registry says " << registryKnown
              << " for wave " << waveId << " block " << blockId << std::endl;
  }

  // Continue validation but keep sync point as authority for now
  bool finalResult = syncPointKnown; // Keep stable behavior

  // Log any discrepancies for monitoring
  if (syncPointKnown != registryKnown) {
    std::cout << "INFO: Participants known - sync point: " << syncPointKnown
              << ", registry: " << registryKnown << " (using sync point)"
              << std::endl;
  }

  if (finalResult) {
    std::cout << "DEBUG: areAllParticipantsKnownForWaveInstruction - All "
                 "participants known for sync point"
              << std::endl;
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
    const std::string &instructionType) {
  // Create compound key with current block ID
  uint32_t blockId = getCurrentBlock(waveId, laneId);
  std::pair<const void *, uint32_t> instructionKey = {instruction, blockId};

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
  auto* block = getBlock(blockId);
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
  const auto* block = getBlock(blockId);
  if (block) {
    return block->getInstructionList();
  }
  return {};
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getInstructionParticipantsInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  const auto* block = getBlock(blockId);
  if (block) {
    const auto &instructionParticipants =
        block->getInstructionParticipants();
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
  const auto* block = getBlock(blockId);
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
  const auto* block = getBlock(blockId);
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
  std::cout << "DEBUG: createIfBlocks - ifStmt=" << ifStmt
            << ", parentBlockId=" << parentBlockId << ", hasElse=" << hasElse
            << std::endl;
  std::cout << "DEBUG: createIfBlocks - mergeStack size=" << mergeStack.size()
            << std::endl;
  for (size_t i = 0; i < mergeStack.size(); i++) {
    std::cout << "  MergeStack[" << i
              << "]: sourceStatement=" << mergeStack[i].sourceStatement
              << std::endl;
  }
  std::cout << "DEBUG: createIfBlocks - executionPath size="
            << executionPath.size() << std::endl;
  for (size_t i = 0; i < executionPath.size(); i++) {
    std::cout << "  ExecutionPath[" << i << "]=" << executionPath[i]
              << std::endl;
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

  std::cout << "DEBUG: createIfBlocks - Created blocks: thenBlockId="
            << thenBlockId << ", elseBlockId=" << elseBlockId
            << ", mergeBlockId=" << mergeBlockId << std::endl;

  return {thenBlockId, elseBlockId, mergeBlockId};
}

void ThreadgroupContext::moveThreadFromUnknownToParticipating(uint32_t blockId,
                                                              WaveId waveId,
                                                              LaneId laneId) {
  auto* block = getBlock(blockId);
  if (!block)
    return;
  // Update lane assignment
  std::cout << "DEBUG: moveThreadFromUnknownToParticipating - moving lane "
            << laneId << " to block " << blockId << std::endl;
  assignLaneToBlock(waveId, laneId, blockId);

  // Verify the assignment worked
  uint32_t newBlock = getCurrentBlock(waveId, laneId);
  std::cout << "DEBUG: moveThreadFromUnknownToParticipating - lane " << laneId
            << " is now in block " << newBlock << std::endl;
}

void ThreadgroupContext::removeThreadFromUnknown(uint32_t blockId,
                                                 WaveId waveId, LaneId laneId) {
  auto* block = getBlock(blockId);
  if (!block)
    return;

  // Remove lane from unknown (it chose a different path)
  std::cout << "DEBUG: removeThreadFromUnknown - removing lane " << laneId
            << " from block " << blockId << std::endl;

  // Show unknown lanes before removal
  auto unknownBefore = membershipRegistry.getUnknownLanes(waveId, blockId);
  std::cout << "DEBUG: Block " << blockId << " unknown lanes before removal: {";
  for (auto it = unknownBefore.begin(); it != unknownBefore.end(); ++it) {
    if (it != unknownBefore.begin())
      std::cout << " ";
    std::cout << *it;
  }
  std::cout << "}" << std::endl;

  // Update both old system and registry for consistency
  // block.removeUnknownLane(waveId, laneId);
  membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                   LaneBlockStatus::Left);

  // Show unknown lanes after removal
  auto unknownAfter = membershipRegistry.getUnknownLanes(waveId, blockId);
  std::cout << "DEBUG: Block " << blockId << " unknown lanes after removal: {";
  for (auto it = unknownAfter.begin(); it != unknownAfter.end(); ++it) {
    if (it != unknownAfter.begin())
      std::cout << " ";
    std::cout << *it;
  }
  std::cout << "}" << std::endl;

  // Update resolution status
  auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
  bool allResolved = unknownLanes.empty();
  std::cout << "DEBUG: Block " << blockId
            << " areAllUnknownLanesResolvedForWave(" << waveId
            << ") = " << allResolved << " (tracked by registry)" << std::endl;
  // Resolution status tracked by registry - no need for old system metadata

  // CRITICAL: Check if removing this unknown lane allows any waiting wave
  // operations to proceed This handles the case where lanes 0,1 are waiting in
  // block 2 for a wave op, and lanes 2,3 choosing the else branch resolves all
  // unknowns
  if (unknownLanes.empty()) {
    std::cout << "DEBUG: Block " << blockId
              << " now has all unknowns resolved for wave " << waveId
              << " - checking for ready wave operations" << std::endl;

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
          std::cout << "DEBUG: Checking if waiting lane " << waitingLaneId
                    << " can now execute wave operation in block " << blockId
                    << std::endl;

          // Re-evaluate the sync point for this instruction
          auto &syncPoint = waves[waveId]->activeSyncPoints[instructionKey];
          // Sync point state is now computed on-demand

          if (syncPoint.isAllParticipantsArrived() &&
              syncPoint.isAllParticipantsKnown(*this, waveId)) {
            std::cout << "DEBUG: Wave operation in block " << blockId
                      << " is now ready to execute - all participants known "
                         "and arrived!"
                      << std::endl;
          }
        }
      }
    }
  }
}

// Helper method to completely remove a lane from all sets of a specific block
void ThreadgroupContext::removeThreadFromAllSets(uint32_t blockId,
                                                 WaveId waveId, LaneId laneId) {

  const auto* block = getBlock(blockId);
  if (block) {
    std::cout << "DEBUG: removeThreadFromAllSets - removing lane " << laneId
              << " from all sets of block " << blockId << std::endl;

    // Also update the BlockMembershipRegistry
    auto partLanesBefore =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    auto waitingLanesBefore =
        membershipRegistry.getWaitingLanes(waveId, blockId);
    size_t laneCountBefore = partLanesBefore.size() + waitingLanesBefore.size();

    // Show which lanes are in the block before removal
    std::cout << "DEBUG: removeThreadFromAllSets - block " << blockId << " had "
              << laneCountBefore << " participating lanes before removal";
    if (!partLanesBefore.empty()) {
      std::cout << " (participating: ";
      for (auto lid : partLanesBefore) {
        std::cout << lid << " ";
      }
      std::cout << ")";
    }
    if (!waitingLanesBefore.empty()) {
      std::cout << " (waiting: ";
      for (auto lid : waitingLanesBefore) {
        std::cout << lid << " ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;

    membershipRegistry.setLaneStatus(waveId, laneId, blockId,
                                     LaneBlockStatus::Left);

    auto partLanesAfter =
        membershipRegistry.getParticipatingLanes(waveId, blockId);
    auto waitingLanesAfter =
        membershipRegistry.getWaitingLanes(waveId, blockId);
    size_t laneCountAfter = partLanesAfter.size() + waitingLanesAfter.size();

    // Show which lanes are in the block after removal
    std::cout << "DEBUG: removeThreadFromAllSets - block " << blockId << " has "
              << laneCountAfter << " participating lanes after removal";
    if (!partLanesAfter.empty()) {
      std::cout << " (participating: ";
      for (auto lid : partLanesAfter) {
        std::cout << lid << " ";
      }
      std::cout << ")";
    }
    if (!waitingLanesAfter.empty()) {
      std::cout << " (waiting: ";
      for (auto lid : waitingLanesAfter) {
        std::cout << lid << " ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;
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
        std::cout << "DEBUG: removeThreadFromNestedBlocks - skipping "
                  << blockId << " (LOOP_EXIT block where lanes should go)"
                  << std::endl;
        continue;
      }

      // This is a direct child of the parent block - remove from all sets
      std::cout << "DEBUG: removeThreadFromNestedBlocks - removing lane "
                << laneId << " from block " << blockId << " (child of "
                << parentBlockId << ")" << std::endl;
      removeThreadFromAllSets(blockId, waveId, laneId);

      // Recursively remove from nested blocks of this child
      removeThreadFromNestedBlocks(blockId, waveId, laneId);
    }
  }
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getCurrentBlockParticipants(uint32_t blockId) const {
  const auto* block = getBlock(blockId);
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
  INTERPRETER_DEBUG_LOG(
      "\n=== Dynamic Execution Graph (MiniHLSL Interpreter) ===\n");
  INTERPRETER_DEBUG_LOG("Threadgroup Size: " << threadgroupSize << "\n");
  INTERPRETER_DEBUG_LOG("Wave Size: " << waveSize << "\n");
  INTERPRETER_DEBUG_LOG("Wave Count: " << waveCount << "\n");
  INTERPRETER_DEBUG_LOG("Total Dynamic Blocks: " << executionBlocks.size()
                                                 << "\n");
  INTERPRETER_DEBUG_LOG("Next Block ID: " << nextBlockId << "\n\n");

  // Print each dynamic execution block
  for (const auto &[blockId, block] : executionBlocks) {
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

void ThreadgroupContext::printBlockDetails(uint32_t blockId,
                                           bool verbose) const {
  const auto* block = getBlock(blockId);
  if (!block) {
    INTERPRETER_DEBUG_LOG("Block " << blockId << ": NOT FOUND\n");
    return;
  }
  INTERPRETER_DEBUG_LOG("Dynamic Block " << blockId << ":\n");

  // Basic block info
  INTERPRETER_DEBUG_LOG("  Block ID: " << block->getBlockId() << "\n");

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
  case BlockType::LOOP_BODY:
    blockTypeName = "LOOP_BODY";
    break;
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
  INTERPRETER_DEBUG_LOG("  Block Type: " << blockTypeName << "\n");
  INTERPRETER_DEBUG_LOG("  Parent Block: " << block->getParentBlockId() << "\n");
  INTERPRETER_DEBUG_LOG("  Program Point: " << block->getProgramPoint() << "\n");
  INTERPRETER_DEBUG_LOG(
      "  Is Converged: " << (block->getIsConverged() ? "Yes" : "No") << "\n");
  INTERPRETER_DEBUG_LOG("  Nesting Level: " << block->getNestingLevel() << "\n");

  // Source statement info
  if (block->getSourceStatement()) {
    INTERPRETER_DEBUG_LOG(
        "  Source Statement: "
        << static_cast<const void *>(block->getSourceStatement()) << "\n");
  }

  // Participating lanes by wave
  size_t totalLanes = 0;
  INTERPRETER_DEBUG_LOG("  Participating Lanes by Wave:\n");
  for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
    auto laneSet = membershipRegistry.getParticipatingLanes(waveId, blockId);
    INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
    bool first = true;
    for (LaneId laneId : laneSet) {
      if (!first)
        INTERPRETER_DEBUG_LOG(", ");
      INTERPRETER_DEBUG_LOG(laneId);
      first = false;
    }
    INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
    totalLanes += laneSet.size();
  }
  INTERPRETER_DEBUG_LOG("  Total Participating Lanes: " << totalLanes << "\n");

  if (verbose) {
    // Unknown lanes
    bool hasUnknownLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto unknownLanes = membershipRegistry.getUnknownLanes(waveId, blockId);
      if (!unknownLanes.empty()) {
        if (!hasUnknownLanes) {
          INTERPRETER_DEBUG_LOG("  Unknown Lanes by Wave:\n");
          hasUnknownLanes = true;
        }
        auto laneSet = unknownLanes;
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }

    // Arrived lanes
    bool hasArrivedLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto arrivedLanes = membershipRegistry.getArrivedLanes(waveId, blockId);
      if (!arrivedLanes.empty()) {
        if (!hasArrivedLanes) {
          INTERPRETER_DEBUG_LOG("  Arrived Lanes by Wave:\n");
          hasArrivedLanes = true;
        }
        auto laneSet = arrivedLanes;
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }

    // Waiting lanes
    bool hasWaitingLanes = false;
    for (WaveId waveId = 0; waveId < waves.size(); ++waveId) {
      auto waitingLanes = membershipRegistry.getWaitingLanes(waveId, blockId);
      if (!waitingLanes.empty()) {
        if (!hasWaitingLanes) {
          INTERPRETER_DEBUG_LOG("  Waiting Lanes by Wave:\n");
          hasWaitingLanes = true;
        }
        auto laneSet = waitingLanes;
        INTERPRETER_DEBUG_LOG("    Wave " << waveId << ": {");
        bool first = true;
        for (LaneId laneId : laneSet) {
          if (!first)
            INTERPRETER_DEBUG_LOG(", ");
          INTERPRETER_DEBUG_LOG(laneId);
          first = false;
        }
        INTERPRETER_DEBUG_LOG("} (" << laneSet.size() << " lanes)\n");
      }
    }

    // Instructions in this block
    const auto &instructions = block->getInstructionList();
    if (!instructions.empty()) {
      INTERPRETER_DEBUG_LOG("  Instructions (" << instructions.size()
                                               << "):\n");
      for (size_t i = 0; i < instructions.size(); ++i) {
        const auto &instr = instructions[i];
        INTERPRETER_DEBUG_LOG("    " << i << ": " << instr.instructionType
                                     << " (ptr: " << instr.instruction
                                     << ")\n");
      }
    }
  }
}

void ThreadgroupContext::printWaveState(WaveId waveId, bool verbose) const {
  if (waveId >= waves.size()) {
    INTERPRETER_DEBUG_LOG("Wave " << waveId << ": NOT FOUND\n");
    return;
  }

  const auto &wave = waves[waveId];
  INTERPRETER_DEBUG_LOG("Wave " << waveId << ":\n");
  INTERPRETER_DEBUG_LOG("  Wave Size: " << wave->waveSize << "\n");
  INTERPRETER_DEBUG_LOG("  Lane Count: " << wave->lanes.size() << "\n");
  INTERPRETER_DEBUG_LOG("  Active Lanes: " << wave->countActiveLanes() << "\n");
  INTERPRETER_DEBUG_LOG("  Currently Active Lanes: "
                        << wave->countCurrentlyActiveLanes() << "\n");

  if (verbose) {
    // Lane to block mapping
    INTERPRETER_DEBUG_LOG("  Lane to Block Mapping (from registry):\n");
    for (LaneId laneId = 0; laneId < wave->lanes.size(); ++laneId) {
      uint32_t blockId = getCurrentBlock(waveId, laneId);
      if (blockId != 0) {
        INTERPRETER_DEBUG_LOG("    Lane " << laneId << " -> Block " << blockId
                                          << "\n");
      }
    }

    // Active sync points
    if (!wave->activeSyncPoints.empty()) {
      INTERPRETER_DEBUG_LOG("  Active Sync Points ("
                            << wave->activeSyncPoints.size() << "):\n");
      for (const auto &[instructionKey, syncPoint] : wave->activeSyncPoints) {
        INTERPRETER_DEBUG_LOG("    Instruction "
                              << instructionKey.first << " block "
                              << instructionKey.second << " ("
                              << syncPoint.instructionType << "):\n");
        INTERPRETER_DEBUG_LOG("      Expected: "
                              << syncPoint.expectedParticipants.size()
                              << " lanes\n");
        INTERPRETER_DEBUG_LOG("      Arrived: "
                              << syncPoint.arrivedParticipants.size()
                              << " lanes\n");
        INTERPRETER_DEBUG_LOG(
            "      Ready to execute: "
            << (syncPoint.isReadyToExecute(*this, waveId) ? "Yes" : "No")
            << "\n");
      }
    }
  }
}

std::string ThreadgroupContext::getBlockSummary(uint32_t blockId) const {
  const auto* block = getBlock(blockId);
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
DynamicExecutionBlock* ThreadgroupContext::getBlock(uint32_t blockId) {
  auto it = executionBlocks.find(blockId);
  return (it != executionBlocks.end()) ? &it->second : nullptr;
}

const DynamicExecutionBlock* ThreadgroupContext::getBlock(uint32_t blockId) const {
  auto it = executionBlocks.find(blockId);
  return (it != executionBlocks.end()) ? &it->second : nullptr;
}

void ThreadgroupContext::printFinalVariableValues() const {
  INTERPRETER_DEBUG_LOG("\n=== Final Variable Values ===\n");

  for (size_t waveId = 0; waveId < waves.size(); ++waveId) {
    const auto &wave = waves[waveId];
    INTERPRETER_DEBUG_LOG("Wave " << waveId << ":\n");

    for (size_t laneId = 0; laneId < wave->lanes.size(); ++laneId) {
      const auto &lane = wave->lanes[laneId];
      INTERPRETER_DEBUG_LOG("  Lane " << laneId << ":\n");

      // Print all variables for this lane
      if (lane->variables.empty()) {
        INTERPRETER_DEBUG_LOG("    (no variables)\n");
      } else {
        for (const auto &[varName, value] : lane->variables) {
          INTERPRETER_DEBUG_LOG("    " << varName << " = " << value.toString()
                                       << "\n");
        }
      }

      // Print return value if present
      if (lane->hasReturned) {
        INTERPRETER_DEBUG_LOG("    (returned: " << lane->returnValue.toString()
                                                << ")\n");
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