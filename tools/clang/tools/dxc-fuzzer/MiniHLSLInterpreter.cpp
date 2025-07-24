#include "MiniHLSLInterpreter.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>

// Clang AST includes for conversion
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/OperatorKinds.h"

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
}

ThreadId ThreadgroupContext::getGlobalThreadId(WaveId wid, LaneId lid) const {
    return wid * waveSize + lid;
}

std::pair<WaveId, LaneId> ThreadgroupContext::getWaveAndLane(ThreadId tid) const {
    WaveId wid = tid / waveSize;
    LaneId lid = tid % waveSize;
    return {wid, lid};
}

std::vector<ThreadId> ThreadgroupContext::getReadyThreads() const {
    std::vector<ThreadId> ready;
    for (ThreadId tid = 0; tid < threadgroupSize; ++tid) {
        auto [waveId, laneId] = getWaveAndLane(tid);
        if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
            const auto& lane = waves[waveId]->lanes[laneId];
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
            const auto& lane = waves[waveId]->lanes[laneId];
            if (lane->state == ThreadState::WaitingAtBarrier || 
                lane->state == ThreadState::WaitingForWave) {
                waiting.push_back(tid);
            }
        }
    }
    return waiting;
}

bool ThreadgroupContext::canExecuteWaveOp(WaveId waveId, const std::set<LaneId>& activeLanes) const {
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
        try {
            // Execute body
            for (auto& stmt : body_) {
                stmt->execute(lane, wave, tg);
                if (lane.hasReturned) return;
            }
        } catch (const ControlFlowException& e) {
            if (e.type == ControlFlowException::Break) {
                break; // Exit the loop
            } else if (e.type == ControlFlowException::Continue) {
                // Continue to increment
            }
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
    // Phase 1: Simple barrier implementation
    // TODO: Add proper cooperative barrier handling in Phase 2
    ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
    
    // For now, barriers are no-ops since we don't have full cooperative scheduling
    // In a real GPU, this would synchronize all threads in the threadgroup
}

ExprStmt::ExprStmt(std::unique_ptr<Expression> expr) : expr_(std::move(expr)) {}

void ExprStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
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

// MiniHLSLInterpreter implementation with cooperative scheduling
ExecutionResult MiniHLSLInterpreter::executeWithOrdering(const Program& program, 
                                                       const ThreadOrdering& ordering) {
    ExecutionResult result;
    
    // Create threadgroup context
    const uint32_t waveSize = 32; // Standard wave size
    ThreadgroupContext tgContext(program.getTotalThreads(), waveSize);
    
    try {
        uint32_t orderingIndex = 0;
        uint32_t maxIterations = program.getTotalThreads() * program.statements.size() * 10; // Safety limit
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
                for (const auto& wave : tgContext.waves) {
                    for (const auto& lane : wave->lanes) {
                        if (lane->state != ThreadState::Completed && lane->state != ThreadState::Error) {
                            allCompleted = false;
                            break;
                        }
                    }
                    if (!allCompleted) break;
                }
                
                if (allCompleted) {
                    break; // All threads finished
                }
                
                // Check for deadlock
                auto waitingThreads = tgContext.getWaitingThreads();
                if (!waitingThreads.empty()) {
                    result.errorMessage = "Deadlock detected: threads waiting but no progress possible";
                    break;
                }
                
                continue; // Wait for synchronization to complete
            }
            
            // Select next thread to execute according to ordering
            ThreadId nextTid = selectNextThread(readyThreads, ordering, orderingIndex);
            
            // Execute one step of the selected thread  
            bool continueExecution = executeOneStep(nextTid, program, tgContext);
            if (!continueExecution) {
                break; // Fatal error occurred
            }
        }
        
        if (iteration >= maxIterations) {
            result.errorMessage = "Execution timeout: possible infinite loop or deadlock";
        }
        
        // Collect return values in thread order
        for (ThreadId tid = 0; tid < program.getTotalThreads(); ++tid) {
            auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
            if (waveId < tgContext.waves.size() && laneId < tgContext.waves[waveId]->lanes.size()) {
                result.threadReturnValues.push_back(tgContext.waves[waveId]->lanes[laneId]->returnValue);
            }
        }
        
    } catch (const std::exception& e) {
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
bool MiniHLSLInterpreter::executeOneStep(ThreadId tid, const Program& program, ThreadgroupContext& tgContext) {
    auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
    if (waveId >= tgContext.waves.size()) return false;
    
    auto& wave = *tgContext.waves[waveId];
    if (laneId >= wave.lanes.size()) return false;
    
    auto& lane = *wave.lanes[laneId];
    
    // Check if thread is ready to execute
    if (lane.state != ThreadState::Ready) return true;
    
    // Check if we have more statements to execute
    if (lane.currentStatement >= program.statements.size()) {
        lane.state = ThreadState::Completed;
        return true;
    }
    
    // Execute the current statement
    try {
        const auto& stmt = program.statements[lane.currentStatement];
        stmt->execute(lane, wave, tgContext);
        lane.currentStatement++;
        
        if (lane.hasReturned) {
            lane.state = ThreadState::Completed;
        }
    } catch (const std::exception& e) {
        lane.state = ThreadState::Error;
        lane.errorMessage = e.what();
    }
    
    return true;
}

void MiniHLSLInterpreter::processWaveOperations(ThreadgroupContext& tgContext) {
    // Phase 1: Simple implementation - wave ops complete immediately
    // TODO: Add proper cooperative scheduling in Phase 2
}

void MiniHLSLInterpreter::processBarriers(ThreadgroupContext& tgContext) {
    // Phase 1: Simple implementation - barriers complete immediately  
    // TODO: Add proper barrier analysis in Phase 2
}

ThreadId MiniHLSLInterpreter::selectNextThread(const std::vector<ThreadId>& readyThreads, 
                                              const ThreadOrdering& ordering, 
                                              uint32_t& orderingIndex) {
    // Simple round-robin selection from ready threads
    // Try to follow the ordering preference when possible
    if (readyThreads.empty()) return 0;
    
    // Look for the next thread in ordering that's ready
    for (uint32_t i = 0; i < ordering.executionOrder.size(); ++i) {
        uint32_t idx = (orderingIndex + i) % ordering.executionOrder.size();
        ThreadId tid = ordering.executionOrder[idx];
        
        if (std::find(readyThreads.begin(), readyThreads.end(), tid) != readyThreads.end()) {
            orderingIndex = (idx + 1) % ordering.executionOrder.size();
            return tid;
        }
    }
    
    // Fallback to first ready thread
    return readyThreads[0];
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

// HLSL AST conversion implementation (simplified version)
MiniHLSLInterpreter::ConversionResult 
MiniHLSLInterpreter::convertFromHLSLAST(const clang::FunctionDecl* func, clang::ASTContext& context) {
    ConversionResult result;
    result.success = false;
    
    if (!func || !func->hasBody()) {
        result.errorMessage = "Function has no body or is null";
        return result;
    }
    
    std::cout << "Converting HLSL function: " << func->getName().str() << std::endl;
    
    try {
        // Extract thread configuration from function attributes
        extractThreadConfiguration(func, result.program);
        
        // Get the function body
        const clang::CompoundStmt* body = clang::dyn_cast<clang::CompoundStmt>(func->getBody());
        if (!body) {
            result.errorMessage = "Function body is not a compound statement";
            return result;
        }
        
        // Convert the function body to interpreter statements
        convertCompoundStatement(body, result.program, context);
        
        std::cout << "Converted AST to interpreter program with " << result.program.statements.size() << " statements" << std::endl;
        
        result.success = true;
        return result;
        
    } catch (const std::exception& e) {
        result.errorMessage = std::string("Exception during conversion: ") + e.what();
        return result;
    }
}

// AST traversal helper methods
void MiniHLSLInterpreter::extractThreadConfiguration(const clang::FunctionDecl* func, Program& program) {
    // Default configuration
    program.numThreadsX = 1;
    program.numThreadsY = 1; 
    program.numThreadsZ = 1;
    
    // Look for HLSLNumThreadsAttr
    if (const clang::HLSLNumThreadsAttr *attr = func->getAttr<clang::HLSLNumThreadsAttr>()) {
        program.numThreadsX = attr->getX();
        program.numThreadsY = attr->getY();
        program.numThreadsZ = attr->getZ();
        std::cout << "Found numthreads attribute: [" << program.numThreadsX 
                  << ", " << program.numThreadsY << ", " << program.numThreadsZ << "]" << std::endl;
    } else {
        std::cout << "No numthreads attribute found, using default [1, 1, 1]" << std::endl;
    }
}

void MiniHLSLInterpreter::convertCompoundStatement(const clang::CompoundStmt* compound, 
                                                  Program& program, 
                                                  clang::ASTContext& context) {
    std::cout << "Converting compound statement with " << compound->size() << " child statements" << std::endl;
    
    for (const auto* stmt : compound->children()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
            program.statements.push_back(std::move(convertedStmt));
        }
    }
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertStatement(const clang::Stmt* stmt, 
                                                               clang::ASTContext& context) {
    if (!stmt) return nullptr;
    
    std::cout << "Converting statement: " << stmt->getStmtClassName() << std::endl;
    
    // Handle different statement types
    if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(stmt)) {
        return convertBinaryOperator(binOp, context);
    }
    else if (auto callExpr = clang::dyn_cast<clang::CallExpr>(stmt)) {
        return convertCallExpression(callExpr, context);
    }
    else if (auto declStmt = clang::dyn_cast<clang::DeclStmt>(stmt)) {
        return convertDeclarationStatement(declStmt, context);
    }
    // Note: ExprStmt doesn't exist in DXC's Clang fork
    // Expression statements are handled differently or don't exist
    else if (auto ifStmt = clang::dyn_cast<clang::IfStmt>(stmt)) {
        return convertIfStatement(ifStmt, context);
    }
    else if (auto forStmt = clang::dyn_cast<clang::ForStmt>(stmt)) {
        return convertForStatement(forStmt, context);
    }
    else if (auto whileStmt = clang::dyn_cast<clang::WhileStmt>(stmt)) {
        return convertWhileStatement(whileStmt, context);
    }
    else if (auto doStmt = clang::dyn_cast<clang::DoStmt>(stmt)) {
        return convertDoStatement(doStmt, context);
    }
    else if (auto switchStmt = clang::dyn_cast<clang::SwitchStmt>(stmt)) {
        return convertSwitchStatement(switchStmt, context);
    }
    else if (auto breakStmt = clang::dyn_cast<clang::BreakStmt>(stmt)) {
        return convertBreakStatement(breakStmt, context);
    }
    else if (auto continueStmt = clang::dyn_cast<clang::ContinueStmt>(stmt)) {
        return convertContinueStatement(continueStmt, context);
    }
    else if (auto compound = clang::dyn_cast<clang::CompoundStmt>(stmt)) {
        // Nested compound statement - this should not happen in our current design
        std::cout << "Warning: nested compound statement found, skipping" << std::endl;
        return nullptr;
    }
    else {
        std::cout << "Unsupported statement type: " << stmt->getStmtClassName() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertBinaryOperator(const clang::BinaryOperator* binOp,
                                                                     clang::ASTContext& context) {
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertCallExpression(const clang::CallExpr* callExpr,
                                                                     clang::ASTContext& context) {
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
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Sum));
            }
        }
        else if (funcName == "WaveActiveProduct" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Product));
            }
        }
        else if (funcName == "WaveActiveMin" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Min));
            }
        }
        else if (funcName == "WaveActiveMax" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Max));
            }
        }
        else if (funcName == "WaveActiveAnd" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::And));
            }
        }
        else if (funcName == "WaveActiveOr" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or));
            }
        }
        else if (funcName == "WaveActiveXor" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Xor));
            }
        }
        else if (funcName == "WaveActiveCountBits" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::CountBits));
            }
        }
        else if (funcName == "WaveGetLaneIndex" && callExpr->getNumArgs() == 0) {
            return std::make_unique<ExprStmt>(std::make_unique<LaneIndexExpr>());
        }
        else if (funcName == "WaveGetLaneCount" && callExpr->getNumArgs() == 0) {
            return std::make_unique<ExprStmt>(std::make_unique<WaveGetLaneCountExpr>());
        }
        
        // Handle other function calls as needed
        std::cout << "Unsupported function call: " << funcName << std::endl;
    }
    
    return nullptr;
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertDeclarationStatement(const clang::DeclStmt* declStmt,
                                                                           clang::ASTContext& context) {
    std::cout << "Converting declaration statement" << std::endl;
    
    for (const auto* decl : declStmt->decls()) {
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertIfStatement(const clang::IfStmt* ifStmt,
                                                                  clang::ASTContext& context) {
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
    
    return std::make_unique<IfStmt>(std::move(condition), std::move(thenBlock), std::move(elseBlock));
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertForStatement(const clang::ForStmt* forStmt,
                                                                   clang::ASTContext& context) {
    std::cout << "Converting for statement" << std::endl;
    
    // For loops in HLSL typically have the structure: for (init; condition; increment) { body }
    // We need to extract each component
    
    // Extract the loop variable from the init statement
    std::string loopVar;
    std::unique_ptr<Expression> init = nullptr;
    
    if (auto initStmt = forStmt->getInit()) {
        if (auto declStmt = clang::dyn_cast<clang::DeclStmt>(initStmt)) {
            // Handle variable declaration: int i = 0
            for (const auto* decl : declStmt->decls()) {
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
                 << ", increment: " << (increment ? "yes" : "no") << ")" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<ForStmt>(loopVar, std::move(init), std::move(condition), 
                                    std::move(increment), std::move(body));
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertWhileStatement(const clang::WhileStmt* whileStmt,
                                                                     clang::ASTContext& context) {
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertDoStatement(const clang::DoStmt* doStmt,
                                                                  clang::ASTContext& context) {
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

std::unique_ptr<Statement> MiniHLSLInterpreter::convertSwitchStatement(const clang::SwitchStmt* switchStmt,
                                                                      clang::ASTContext& context) {
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
                            switchResult->addCase(currentCaseValue.value(), std::move(currentCase));
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
                            switchResult->addCase(currentCaseValue.value(), std::move(currentCase));
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
                    switchResult->addCase(currentCaseValue.value(), std::move(currentCase));
                }
            }
        }
    }
    
    return switchResult;
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertBreakStatement(const clang::BreakStmt* breakStmt,
                                                                     clang::ASTContext& context) {
    std::cout << "Converting break statement" << std::endl;
    return std::make_unique<BreakStmt>();
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertContinueStatement(const clang::ContinueStmt* continueStmt,
                                                                        clang::ASTContext& context) {
    std::cout << "Converting continue statement" << std::endl;
    return std::make_unique<ContinueStmt>();
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertExpression(const clang::Expr* expr, 
                                                                  clang::ASTContext& context) {
    if (!expr) return nullptr;
    
    std::cout << "Converting expression: " << expr->getStmtClassName() << std::endl;
    
    // Handle different expression types
    if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(expr)) {
        return convertBinaryExpression(binOp, context);
    }
    else if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) {
        std::string varName = declRef->getDecl()->getName().str();
        return makeVariable(varName);
    }
    else if (auto intLit = clang::dyn_cast<clang::IntegerLiteral>(expr)) {
        int64_t value = intLit->getValue().getSExtValue();
        return makeLiteral(Value(static_cast<int>(value)));
    }
    else if (auto floatLit = clang::dyn_cast<clang::FloatingLiteral>(expr)) {
        double value = floatLit->getValueAsApproximateDouble();
        return makeLiteral(Value(static_cast<float>(value)));
    }
    else if (auto boolLit = clang::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
        bool value = boolLit->getValue();
        return makeLiteral(Value(value));
    }
    else if (auto parenExpr = clang::dyn_cast<clang::ParenExpr>(expr)) {
        return convertExpression(parenExpr->getSubExpr(), context);
    }
    else if (auto implicitCast = clang::dyn_cast<clang::ImplicitCastExpr>(expr)) {
        return convertExpression(implicitCast->getSubExpr(), context);
    }
    else if (auto operatorCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        return convertOperatorCall(operatorCall, context);
    }
    else if (auto callExpr = clang::dyn_cast<clang::CallExpr>(expr)) {
        return convertCallExpressionToExpression(callExpr, context);
    }
    else if (auto condOp = clang::dyn_cast<clang::ConditionalOperator>(expr)) {
        return convertConditionalOperator(condOp, context);
    }
    else {
        std::cout << "Unsupported expression type: " << expr->getStmtClassName() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertCallExpressionToExpression(const clang::CallExpr* callExpr,
                                                                                 clang::ASTContext& context) {
    if (auto funcDecl = callExpr->getDirectCallee()) {
        std::string funcName = funcDecl->getName().str();
        std::cout << "Converting function call to expression: " << funcName << std::endl;
        
        // Check for wave intrinsic functions that return values
        if (funcName == "WaveActiveSum" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Sum);
            }
        }
        else if (funcName == "WaveActiveProduct" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Product);
            }
        }
        else if (funcName == "WaveActiveMin" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Min);
            }
        }
        else if (funcName == "WaveActiveMax" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Max);
            }
        }
        else if (funcName == "WaveActiveAnd" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::And);
            }
        }
        else if (funcName == "WaveActiveOr" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or);
            }
        }
        else if (funcName == "WaveActiveXor" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Xor);
            }
        }
        else if (funcName == "WaveActiveCountBits" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::CountBits);
            }
        }
        else if (funcName == "WaveGetLaneIndex" && callExpr->getNumArgs() == 0) {
            return std::make_unique<LaneIndexExpr>();
        }
        else if (funcName == "WaveGetLaneCount" && callExpr->getNumArgs() == 0) {
            return std::make_unique<WaveGetLaneCountExpr>();
        }
        else if (funcName == "WaveIsFirstLane" && callExpr->getNumArgs() == 0) {
            return std::make_unique<WaveIsFirstLaneExpr>();
        }
        else if (funcName == "WaveActiveAllEqual" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveAllEqualExpr>(std::move(arg));
            }
        }
        else if (funcName == "WaveActiveAllTrue" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveAllTrueExpr>(std::move(arg));
            }
        }
        else if (funcName == "WaveActiveAnyTrue" && callExpr->getNumArgs() == 1) {
            auto arg = convertExpression(callExpr->getArg(0), context);
            if (arg) {
                return std::make_unique<WaveActiveAnyTrueExpr>(std::move(arg));
            }
        }
        
        std::cout << "Unsupported function call in expression context: " << funcName << std::endl;
    }
    
    return nullptr;
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertConditionalOperator(const clang::ConditionalOperator* condOp,
                                                                          clang::ASTContext& context) {
    std::cout << "Converting conditional operator (ternary)" << std::endl;
    
    // Convert the condition
    auto condition = convertExpression(condOp->getCond(), context);
    if (!condition) {
        std::cout << "Failed to convert conditional operator condition" << std::endl;
        return nullptr;
    }
    
    // Convert the true expression
    auto trueExpr = convertExpression(condOp->getTrueExpr(), context);
    if (!trueExpr) {
        std::cout << "Failed to convert conditional operator true expression" << std::endl;
        return nullptr;
    }
    
    // Convert the false expression
    auto falseExpr = convertExpression(condOp->getFalseExpr(), context);
    if (!falseExpr) {
        std::cout << "Failed to convert conditional operator false expression" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<ConditionalExpr>(std::move(condition), std::move(trueExpr), std::move(falseExpr));
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertBinaryExpression(const clang::BinaryOperator* binOp,
                                                                        clang::ASTContext& context) {
    auto lhs = convertExpression(binOp->getLHS(), context);
    auto rhs = convertExpression(binOp->getRHS(), context);
    
    if (!lhs || !rhs) return nullptr;
    
    // Map Clang binary operator to interpreter binary operator
    BinaryOpExpr::OpType opType;
    switch (binOp->getOpcode()) {
        case clang::BO_Add: opType = BinaryOpExpr::Add; break;
        case clang::BO_Sub: opType = BinaryOpExpr::Sub; break; 
        case clang::BO_Mul: opType = BinaryOpExpr::Mul; break;
        case clang::BO_Div: opType = BinaryOpExpr::Div; break;
        case clang::BO_Rem: opType = BinaryOpExpr::Mod; break;
        default:
            std::cout << "Unsupported binary operator" << std::endl;
            return nullptr;
    }
    
    return makeBinaryOp(std::move(lhs), std::move(rhs), opType);
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertOperatorCall(const clang::CXXOperatorCallExpr* opCall,
                                                                    clang::ASTContext& context) {
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
                // For now, we'll use a simple model where each buffer starts at address 0
                // and each element is 4 bytes (size of int/float)
                
                // Create a shared memory read expression
                // Note: In a real implementation, we'd need to track buffer base addresses
                // and handle different data types/sizes
                MemoryAddress baseAddr = 0; // Simplified: assume buffer starts at 0
                
                // Create an expression that computes: baseAddr + index * sizeof(element)
                auto sizeofElement = makeLiteral(Value(4)); // Assume 4 bytes per element
                auto offset = std::make_unique<BinaryOpExpr>(
                    std::move(indexExpr), 
                    std::move(sizeofElement), 
                    BinaryOpExpr::Mul
                );
                
                // For now, just use the index directly as the address (simplified)
                // In a real implementation, we'd add the base address
                return std::make_unique<SharedReadExpr>(0); // Placeholder
                
                // TODO: Properly implement SharedReadExpr that takes a dynamic address expression
                // return std::make_unique<SharedReadExpr>(std::move(offset));
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
    : condition_(std::move(condition)), trueExpr_(std::move(trueExpr)), falseExpr_(std::move(falseExpr)) {}

Value ConditionalExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    if (!lane.isActive) return Value(0);
    
    auto condValue = condition_->evaluate(lane, wave, tg);
    bool cond = condValue.asBool();
    
    if (cond) {
        return trueExpr_->evaluate(lane, wave, tg);
    } else {
        return falseExpr_->evaluate(lane, wave, tg);
    }
}

bool ConditionalExpr::isDeterministic() const {
    return condition_->isDeterministic() && trueExpr_->isDeterministic() && falseExpr_->isDeterministic();
}

std::string ConditionalExpr::toString() const {
    return "(" + condition_->toString() + " ? " + trueExpr_->toString() + " : " + falseExpr_->toString() + ")";
}

// WaveIsFirstLaneExpr implementation
Value WaveIsFirstLaneExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    if (!lane.isActive) return Value(false);
    
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

Value WaveActiveAllEqualExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    if (!lane.isActive) return Value(false);
    
    // Get value from the first active lane
    Value firstValue;
    bool foundFirst = false;
    
    for (auto& otherLane : wave.lanes) {
        if (otherLane->isActive) {
            Value val = expr_->evaluate(*otherLane, wave, tg);
            if (!foundFirst) {
                firstValue = val;
                foundFirst = true;
            } else {
                // Compare with first value
                if (val.asInt() != firstValue.asInt()) {
                    return Value(false);
                }
            }
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

Value WaveActiveAllTrueExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    if (!lane.isActive) return Value(false);
    
    // Check if expression is true for all active lanes
    for (auto& otherLane : wave.lanes) {
        if (otherLane->isActive) {
            Value val = expr_->evaluate(*otherLane, wave, tg);
            if (!val.asBool()) {
                return Value(false);
            }
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

Value WaveActiveAnyTrueExpr::evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const {
    if (!lane.isActive) return Value(false);
    
    // Check if expression is true for any active lane
    for (auto& otherLane : wave.lanes) {
        if (otherLane->isActive) {
            Value val = expr_->evaluate(*otherLane, wave, tg);
            if (val.asBool()) {
                return Value(true);
            }
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

void WhileStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    // Execute while loop with condition checking
    while (true) {
        auto condValue = condition_->evaluate(lane, wave, tg);
        if (!condValue.asBool()) break;
        
        try {
            for (auto& stmt : body_) {
                stmt->execute(lane, wave, tg);
                if (!lane.isActive) return; // Early exit if lane becomes inactive
            }
        } catch (const ControlFlowException& e) {
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
    for (const auto& stmt : body_) {
        result += "  " + stmt->toString() + "\n";
    }
    result += "}";
    return result;
}

// DoWhileStmt implementation
DoWhileStmt::DoWhileStmt(std::vector<std::unique_ptr<Statement>> body,
                         std::unique_ptr<Expression> cond)
    : body_(std::move(body)), condition_(std::move(cond)) {}

void DoWhileStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    // Execute do-while loop - body executes at least once
    do {
        try {
            for (auto& stmt : body_) {
                stmt->execute(lane, wave, tg);
                if (!lane.isActive) return; // Early exit if lane becomes inactive
            }
        } catch (const ControlFlowException& e) {
            if (e.type == ControlFlowException::Break) {
                break; // Exit the loop
            } else if (e.type == ControlFlowException::Continue) {
                // Continue to condition check
            }
        }
        
        auto condValue = condition_->evaluate(lane, wave, tg);
        if (!condValue.asBool()) break;
    } while (true);
}

std::string DoWhileStmt::toString() const {
    std::string result = "do {\n";
    for (const auto& stmt : body_) {
        result += "  " + stmt->toString() + "\n";
    }
    result += "} while (" + condition_->toString() + ");";
    return result;
}

// SwitchStmt implementation
SwitchStmt::SwitchStmt(std::unique_ptr<Expression> cond)
    : condition_(std::move(cond)) {}

void SwitchStmt::addCase(int value, std::vector<std::unique_ptr<Statement>> stmts) {
    cases_.push_back({value, std::move(stmts)});
}

void SwitchStmt::addDefault(std::vector<std::unique_ptr<Statement>> stmts) {
    cases_.push_back({std::nullopt, std::move(stmts)});
}

void SwitchStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    
    // Evaluate switch condition
    auto condValue = condition_->evaluate(lane, wave, tg);
    int switchValue = condValue.asInt();
    
    // Find matching case
    bool foundMatch = false;
    bool fallthrough = false;
    
    for (const auto& caseBlock : cases_) {
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
                for (auto& stmt : caseBlock.statements) {
                    stmt->execute(lane, wave, tg);
                    if (!lane.isActive) return;
                }
                // Continue to next case (fallthrough)
            } catch (const ControlFlowException& e) {
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
    for (const auto& caseBlock : cases_) {
        if (caseBlock.value.has_value()) {
            result += "  case " + std::to_string(caseBlock.value.value()) + ":\n";
        } else {
            result += "  default:\n";
        }
        for (const auto& stmt : caseBlock.statements) {
            result += "    " + stmt->toString() + "\n";
        }
    }
    result += "}";
    return result;
}

// BreakStmt implementation
void BreakStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    throw ControlFlowException(ControlFlowException::Break);
}

// ContinueStmt implementation
void ContinueStmt::execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) {
    if (!lane.isActive) return;
    throw ControlFlowException(ControlFlowException::Continue);
}

} // namespace interpreter
} // namespace minihlsl