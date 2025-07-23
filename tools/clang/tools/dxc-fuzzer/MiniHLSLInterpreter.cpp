#include "MiniHLSLInterpreter.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>

namespace minihlsl {
namespace interpreter {

// Value implementation
Value Value::operator+(const Value& other) const {
    if (std::holds_alternative<int32_t>(data) && std::holds_alternative<int32_t>(other.data)) {
        return Value(std::get<int32_t>(data) + std::get<int32_t>(other.data));
    }
    return Value(asFloat() + other.asFloat());
}

Value Value::operator-(const Value& other) const {
    if (std::holds_alternative<int32_t>(data) && std::holds_alternative<int32_t>(other.data)) {
        return Value(std::get<int32_t>(data) - std::get<int32_t>(other.data));
    }
    return Value(asFloat() - other.asFloat());
}

Value Value::operator*(const Value& other) const {
    if (std::holds_alternative<int32_t>(data) && std::holds_alternative<int32_t>(other.data)) {
        return Value(std::get<int32_t>(data) * std::get<int32_t>(other.data));
    }
    return Value(asFloat() * other.asFloat());
}

Value Value::operator/(const Value& other) const {
    if (std::holds_alternative<int32_t>(data) && std::holds_alternative<int32_t>(other.data)) {
        int32_t divisor = std::get<int32_t>(other.data);
        if (divisor == 0) throw std::runtime_error("Division by zero");
        return Value(std::get<int32_t>(data) / divisor);
    }
    float divisor = other.asFloat();
    if (divisor == 0.0f) throw std::runtime_error("Division by zero");
    return Value(asFloat() / divisor);
}

Value Value::operator%(const Value& other) const {
    int32_t a = asInt();
    int32_t b = other.asInt();
    if (b == 0) throw std::runtime_error("Modulo by zero");
    return Value(a % b);
}

bool Value::operator==(const Value& other) const {
    if (data.index() != other.data.index()) {
        return asFloat() == other.asFloat();
    }
    return data == other.data;
}

bool Value::operator!=(const Value& other) const {
    return !(*this == other);
}

bool Value::operator<(const Value& other) const {
    if (std::holds_alternative<int32_t>(data) && std::holds_alternative<int32_t>(other.data)) {
        return std::get<int32_t>(data) < std::get<int32_t>(other.data);
    }
    return asFloat() < other.asFloat();
}

bool Value::operator<=(const Value& other) const {
    return (*this < other) || (*this == other);
}

bool Value::operator>(const Value& other) const {
    return !(*this <= other);
}

bool Value::operator>=(const Value& other) const {
    return !(*this < other);
}

Value Value::operator&&(const Value& other) const {
    return Value(asBool() && other.asBool());
}

Value Value::operator||(const Value& other) const {
    return Value(asBool() || other.asBool());
}

Value Value::operator!() const {
    return Value(!asBool());
}

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
                       [](const auto& lane) { return lane->isActive; });
}

uint32_t WaveContext::countActiveLanes() const {
    return std::count_if(lanes.begin(), lanes.end(),
                        [](const auto& lane) { return lane->isActive; });
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

bool SharedMemory::hasConflictingAccess(MemoryAddress addr, ThreadId tid1, ThreadId tid2) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = accessHistory_.find(addr);
    if (it == accessHistory_.end()) return false;
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
    
    // Initialize waves
    for (uint32_t w = 0; w < waveCount; ++w) {
        auto wave = std::make_unique<WaveContext>();
        wave->waveId = w;
        wave->waveSize = waveSize;
        
        // Initialize lanes in wave
        uint32_t lanesInWave = std::min(waveSize, threadgroupSize - w * waveSize);
        for (uint32_t l = 0; l < lanesInWave; ++l) {
            auto lane = std::make_unique<LaneContext>();
            lane->laneId = l;
            wave->lanes.push_back(std::move(lane));
        }
        
        waves.push_back(std::move(wave));
    }
    
    sharedMemory = std::make_shared<SharedMemory>();
    currentBarrier = std::make_shared<BarrierState>();
}

ThreadId ThreadgroupContext::getGlobalThreadId(WaveId wid, LaneId lid) const {
    return wid * waveSize + lid;
}

std::pair<WaveId, LaneId> ThreadgroupContext::getWaveAndLane(ThreadId tid) const {
    WaveId wid = tid / waveSize;
    LaneId lid = tid % waveSize;
    return {wid, lid};
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
    std::shuffle(ordering.executionOrder.begin(), ordering.executionOrder.end(), rng);
    
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

ThreadOrdering ThreadOrdering::waveInterleaved(uint32_t threadCount, uint32_t waveSize) {
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
Value VariableExpr::evaluate(LaneContext& lane, WaveContext&, ThreadgroupContext&) const {
    auto it = lane.variables.find(name_);
    if (it == lane.variables.end()) {
        throw std::runtime_error("Undefined variable: " + name_);
    }
    return it->second;
}

Value LaneIndexExpr::evaluate(LaneContext& lane, WaveContext&, ThreadgroupContext&) const {
    return Value(static_cast<int32_t>(lane.laneId));
}

Value WaveIndexExpr::evaluate(LaneContext&, WaveContext& wave, ThreadgroupContext&) const {
    return Value(static_cast<int32_t>(wave.waveId));
}

Value ThreadIndexExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
    return Value(static_cast<int32_t>(tid));
}

BinaryOpExpr::BinaryOpExpr(std::unique_ptr<Expression> left, 
                         std::unique_ptr<Expression> right, 
                         OpType op)
    : left_(std::move(left)), right_(std::move(right)), op_(op) {}

Value BinaryOpExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    Value leftVal = left_->evaluate(lane, wave, tg);
    Value rightVal = right_->evaluate(lane, wave, tg);
    
    switch (op_) {
        case Add: return leftVal + rightVal;
        case Sub: return leftVal - rightVal;
        case Mul: return leftVal * rightVal;
        case Div: return leftVal / rightVal;
        case Mod: return leftVal % rightVal;
        case Eq:  return Value(leftVal == rightVal);
        case Ne:  return Value(leftVal != rightVal);
        case Lt:  return Value(leftVal < rightVal);
        case Le:  return Value(leftVal <= rightVal);
        case Gt:  return Value(leftVal > rightVal);
        case Ge:  return Value(leftVal >= rightVal);
        case And: return leftVal && rightVal;
        case Or:  return leftVal || rightVal;
    }
    throw std::runtime_error("Unknown binary operator");
}

bool BinaryOpExpr::isDeterministic() const {
    return left_->isDeterministic() && right_->isDeterministic();
}

std::string BinaryOpExpr::toString() const {
    static const char* opStrings[] = {
        "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||"
    };
    return "(" + left_->toString() + " " + opStrings[op_] + " " + right_->toString() + ")";
}

UnaryOpExpr::UnaryOpExpr(std::unique_ptr<Expression> expr, OpType op)
    : expr_(std::move(expr)), op_(op) {}

Value UnaryOpExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    Value val = expr_->evaluate(lane, wave, tg);
    
    switch (op_) {
        case Neg: return Value(-val.asFloat());
        case Not: return !val;
    }
    throw std::runtime_error("Unknown unary operator");
}

bool UnaryOpExpr::isDeterministic() const {
    return expr_->isDeterministic();
}

std::string UnaryOpExpr::toString() const {
    static const char* opStrings[] = { "-", "!" };
    return opStrings[op_] + expr_->toString();
}

// Wave operation implementations
WaveActiveOp::WaveActiveOp(std::unique_ptr<Expression> expr, OpType op)
    : expr_(std::move(expr)), op_(op) {}

Value WaveActiveOp::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    // Wave operations require all active lanes to participate
    if (!lane.isActive) {
        throw std::runtime_error("Inactive lane executing wave operation");
    }
    
    std::vector<Value> values;
    for (const auto& otherLane : wave.lanes) {
        if (otherLane->isActive) {
            values.push_back(expr_->evaluate(*otherLane, wave, tg));
        }
    }
    
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
                if (values[i] < minVal) minVal = values[i];
            }
            return minVal;
        }
        case Max: {
            Value maxVal = values[0];
            for (size_t i = 1; i < values.size(); ++i) {
                if (values[i] > maxVal) maxVal = values[i];
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
            for (const auto& val : values) {
                if (val.asBool()) count++;
            }
            return Value(count);
        }
    }
    throw std::runtime_error("Unknown wave operation");
}

std::string WaveActiveOp::toString() const {
    static const char* opNames[] = {
        "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
        "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits"
    };
    return std::string(opNames[op_]) + "(" + expr_->toString() + ")";
}

Value WaveGetLaneCountExpr::evaluate(LaneContext&, WaveContext& wave, ThreadgroupContext&) const {
    return Value(static_cast<int32_t>(wave.countActiveLanes()));
}

// Statement implementations
VarDeclStmt::VarDeclStmt(const std::string& name, std::unique_ptr<Expression> init)
    : name_(name), init_(std::move(init)) {}

void VarDeclStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    Value initVal = init_ ? init_->evaluate(lane, wave, tg) : Value(0);
    lane.variables[name_] = initVal;
}

std::string VarDeclStmt::toString() const {
    return "var " + name_ + " = " + (init_ ? init_->toString() : "0") + ";";
}

AssignStmt::AssignStmt(const std::string& name, std::unique_ptr<Expression> expr)
    : name_(name), expr_(std::move(expr)) {}

void AssignStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    lane.variables[name_] = expr_->evaluate(lane, wave, tg);
}

std::string AssignStmt::toString() const {
    return name_ + " = " + expr_->toString() + ";";
}

IfStmt::IfStmt(std::unique_ptr<Expression> cond, 
               std::vector<std::unique_ptr<Statement>> thenBlock,
               std::vector<std::unique_ptr<Statement>> elseBlock)
    : condition_(std::move(cond)), 
      thenBlock_(std::move(thenBlock)),
      elseBlock_(std::move(elseBlock)) {}

void IfStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    bool condValue = condition_->evaluate(lane, wave, tg).asBool();
    
    // Save current active state
    bool wasActive = lane.isActive;
    
    // Execute then block
    lane.isActive = wasActive && condValue;
    for (auto& stmt : thenBlock_) {
        stmt->execute(lane, wave, tg);
        if (lane.hasReturned) break;
    }
    
    // Execute else block
    if (!elseBlock_.empty() && !lane.hasReturned) {
        lane.isActive = wasActive && !condValue;
        for (auto& stmt : elseBlock_) {
            stmt->execute(lane, wave, tg);
            if (lane.hasReturned) break;
        }
    }
    
    // Restore active state (reconvergence)
    lane.isActive = wasActive && !lane.hasReturned;
}

bool IfStmt::requiresAllLanesActive() const {
    // Check if any statement in branches requires all lanes
    for (const auto& stmt : thenBlock_) {
        if (stmt->requiresAllLanesActive()) return true;
    }
    for (const auto& stmt : elseBlock_) {
        if (stmt->requiresAllLanesActive()) return true;
    }
    return false;
}

std::string IfStmt::toString() const {
    std::string result = "if (" + condition_->toString() + ") {\n";
    for (const auto& stmt : thenBlock_) {
        result += "    " + stmt->toString() + "\n";
    }
    result += "}";
    if (!elseBlock_.empty()) {
        result += " else {\n";
        for (const auto& stmt : elseBlock_) {
            result += "    " + stmt->toString() + "\n";
        }
        result += "}";
    }
    return result;
}

ForStmt::ForStmt(const std::string& var, 
                std::unique_ptr<Expression> init,
                std::unique_ptr<Expression> cond,
                std::unique_ptr<Expression> inc,
                std::vector<std::unique_ptr<Statement>> body)
    : loopVar_(var), init_(std::move(init)), condition_(std::move(cond)), 
      increment_(std::move(inc)), body_(std::move(body)) {}

void ForStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    // Initialize loop variable
    lane.variables[loopVar_] = init_->evaluate(lane, wave, tg);
    
    // Execute loop
    while (condition_->evaluate(lane, wave, tg).asBool()) {
        // Execute body
        for (auto& stmt : body_) {
            stmt->execute(lane, wave, tg);
            if (lane.hasReturned) return;
        }
        
        // Increment
        lane.variables[loopVar_] = increment_->evaluate(lane, wave, tg);
    }
}

std::string ForStmt::toString() const {
    std::string result = "for (" + loopVar_ + " = " + init_->toString() + "; ";
    result += condition_->toString() + "; ";
    result += loopVar_ + " = " + increment_->toString() + ") {\n";
    for (const auto& stmt : body_) {
        result += "    " + stmt->toString() + "\n";
    }
    result += "}";
    return result;
}

ReturnStmt::ReturnStmt(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

void ReturnStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    if (expr_) {
        lane.returnValue = expr_->evaluate(lane, wave, tg);
    }
    lane.hasReturned = true;
    lane.isActive = false;
}

std::string ReturnStmt::toString() const {
    return "return" + (expr_ ? " " + expr_->toString() : "") + ";";
}

void BarrierStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    // Barriers require all threads to participate
    // In our interpreter, we handle this at the orchestration level
    // This is a synchronization point marker
    ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
    tg.currentBarrier->tryArrive(tid);
}

SharedWriteStmt::SharedWriteStmt(MemoryAddress addr, std::unique_ptr<Expression> expr)
    : addr_(addr), expr_(std::move(expr)) {}

void SharedWriteStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    Value value = expr_->evaluate(lane, wave, tg);
    ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
    tg.sharedMemory->write(addr_, value, tid);
}

std::string SharedWriteStmt::toString() const {
    return "g_shared[" + std::to_string(addr_) + "] = " + expr_->toString() + ";";
}

Value SharedReadExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
    return tg.sharedMemory->read(addr_, tid);
}

std::string SharedReadExpr::toString() const {
    return "g_shared[" + std::to_string(addr_) + "]";
}

// MiniHLSLInterpreter implementation
ExecutionResult MiniHLSLInterpreter::executeWithOrdering(const Program& program, 
                                                       const ThreadOrdering& ordering) {
    ExecutionResult result;
    
    // Create threadgroup context
    const uint32_t waveSize = 32; // Standard wave size
    ThreadgroupContext tgContext(program.getTotalThreads(), waveSize);
    
    // Track barrier synchronization
    std::set<ThreadId> threadsAtBarrier;
    std::vector<ThreadId> pendingThreads = ordering.executionOrder;
    std::set<ThreadId> completedThreads;
    
    // Execute threads according to ordering
    while (!pendingThreads.empty()) {
        std::vector<ThreadId> nextBatch;
        
        for (ThreadId tid : pendingThreads) {
            if (completedThreads.count(tid) > 0) continue;
            
            // Check if thread is waiting at barrier
            if (threadsAtBarrier.count(tid) > 0) {
                // Check if all threads have reached barrier
                if (threadsAtBarrier.size() == program.getTotalThreads()) {
                    // Release all threads from barrier
                    threadsAtBarrier.clear();
                } else {
                    // Thread continues waiting
                    nextBatch.push_back(tid);
                    continue;
                }
            }
            
            // Execute one step of this thread
            auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
            auto& wave = *tgContext.waves[waveId];
            auto& lane = *wave.lanes[laneId];
            
            bool reachedBarrier = false;
            
            // Execute statements until barrier or completion
            for (const auto& stmt : program.statements) {
                if (lane.hasReturned) break;
                
                // Check if this is a barrier
                if (dynamic_cast<BarrierStmt*>(stmt.get())) {
                    stmt->execute(lane, wave, tgContext);
                    threadsAtBarrier.insert(tid);
                    reachedBarrier = true;
                    break;
                }
                
                stmt->execute(lane, wave, tgContext);
            }
            
            if (reachedBarrier) {
                nextBatch.push_back(tid);
            } else {
                completedThreads.insert(tid);
                result.threadReturnValues.push_back(lane.returnValue);
            }
        }
        
        pendingThreads = nextBatch;
        
        // Check for deadlock
        if (!pendingThreads.empty() && threadsAtBarrier.size() < program.getTotalThreads() &&
            threadsAtBarrier.size() == pendingThreads.size()) {
            result.errorMessage = "Deadlock detected: Not all threads reached barrier";
            break;
        }
    }
    
    // Collect final state
    result.sharedMemoryState = tgContext.sharedMemory->getSnapshot();
    
    // Collect global variables from first thread (they should all be the same)
    if (!tgContext.waves.empty() && !tgContext.waves[0]->lanes.empty()) {
        result.globalVariables = tgContext.waves[0]->lanes[0]->variables;
    }
    
    return result;
}

void MiniHLSLInterpreter::executeThread(ThreadId tid, 
                                       const Program& program,
                                       ThreadgroupContext& tgContext) {
    auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
    auto& wave = *tgContext.waves[waveId];
    auto& lane = *wave.lanes[laneId];
    
    for (const auto& stmt : program.statements) {
        if (lane.hasReturned) break;
        stmt->execute(lane, wave, tgContext);
    }
}

bool MiniHLSLInterpreter::areResultsEquivalent(const ExecutionResult& r1, 
                                              const ExecutionResult& r2,
                                              double epsilon) {
    // Check error states
    if (!r1.isValid() || !r2.isValid()) {
        return false;
    }
    
    // Check shared memory state
    if (r1.sharedMemoryState.size() != r2.sharedMemoryState.size()) {
        return false;
    }
    
    for (const auto& [addr, val1] : r1.sharedMemoryState) {
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
    
    std::sort(returns1.begin(), returns1.end(), 
              [](const Value& a, const Value& b) { return a.asFloat() < b.asFloat(); });
    std::sort(returns2.begin(), returns2.end(),
              [](const Value& a, const Value& b) { return a.asFloat() < b.asFloat(); });
    
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
MiniHLSLInterpreter::verifyOrderIndependence(const Program& program, uint32_t numOrderings) {
    VerificationResult verification;
    
    // Generate test orderings
    verification.orderings = generateTestOrderings(program.getTotalThreads(), numOrderings);
    
    // Execute with each ordering
    for (const auto& ordering : verification.orderings) {
        verification.results.push_back(executeWithOrdering(program, ordering));
    }
    
    // Check if all results are equivalent
    verification.isOrderIndependent = true;
    if (!verification.results.empty()) {
        const auto& reference = verification.results[0];
        
        for (size_t i = 1; i < verification.results.size(); ++i) {
            if (!areResultsEquivalent(reference, verification.results[i])) {
                verification.isOrderIndependent = false;
                
                // Generate divergence report
                std::stringstream report;
                report << "Order dependence detected!\n";
                report << "Reference ordering: " << verification.orderings[0].description << "\n";
                report << "Divergent ordering: " << verification.orderings[i].description << "\n";
                report << "Differences in shared memory or return values detected.\n";
                
                verification.divergenceReport = report.str();
                break;
            }
        }
    }
    
    return verification;
}

ExecutionResult MiniHLSLInterpreter::execute(const Program& program, 
                                           const ThreadOrdering& ordering) {
    return executeWithOrdering(program, ordering);
}

std::vector<ThreadOrdering> 
MiniHLSLInterpreter::generateTestOrderings(uint32_t threadCount, uint32_t numOrderings) {
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

std::unique_ptr<Expression> makeVariable(const std::string& name) {
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

std::unique_ptr<Statement> makeVarDecl(const std::string& name, 
                                      std::unique_ptr<Expression> init) {
    return std::make_unique<VarDeclStmt>(name, std::move(init));
}

std::unique_ptr<Statement> makeAssign(const std::string& name,
                                     std::unique_ptr<Expression> expr) {
    return std::make_unique<AssignStmt>(name, std::move(expr));
}

std::unique_ptr<Statement> makeIf(std::unique_ptr<Expression> cond,
                                 std::vector<std::unique_ptr<Statement>> thenBlock,
                                 std::vector<std::unique_ptr<Statement>> elseBlock) {
    return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), std::move(elseBlock));
}

} // namespace interpreter
} // namespace minihlsl