#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <random>
#include <functional>
#include <optional>
#include <variant>
#include <cstdint>
#include <string>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace minihlsl {
namespace interpreter {

// Forward declarations
class Expression;
class Statement;
class WaveOp;
class SharedMemoryOp;

// Basic types
using LaneId = uint32_t;
using WaveId = uint32_t;
using ThreadId = uint32_t;
using MemoryAddress = uint32_t;

// Value type that can hold int, float, or bool
struct Value {
    std::variant<int32_t, float, bool> data;
    
    Value() : data(0) {}
    Value(int32_t i) : data(i) {}
    Value(float f) : data(f) {}
    Value(bool b) : data(b) {}
    
    // Arithmetic operations
    Value operator+(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator/(const Value& other) const;
    Value operator%(const Value& other) const;
    
    // Comparison operations
    bool operator==(const Value& other) const;
    bool operator!=(const Value& other) const;
    bool operator<(const Value& other) const;
    bool operator<=(const Value& other) const;
    bool operator>(const Value& other) const;
    bool operator>=(const Value& other) const;
    
    // Logical operations
    Value operator&&(const Value& other) const;
    Value operator||(const Value& other) const;
    Value operator!() const;
    
    // Type conversions
    int32_t asInt() const;
    float asFloat() const;
    bool asBool() const;
    
    std::string toString() const;
};

// Thread execution context
struct LaneContext {
    LaneId laneId;
    std::map<std::string, Value> variables;
    bool isActive = true;
    bool hasReturned = false;
    Value returnValue;
};

// Wave execution context
struct WaveContext {
    WaveId waveId;
    uint32_t waveSize;
    std::vector<std::unique_ptr<LaneContext>> lanes;
    
    // Get active lane mask
    uint64_t getActiveMask() const;
    std::vector<LaneId> getActiveLanes() const;
    bool allLanesActive() const;
    uint32_t countActiveLanes() const;
};

// Shared memory state
class SharedMemory {
private:
    std::map<MemoryAddress, Value> memory_;
    std::map<MemoryAddress, std::set<ThreadId>> accessHistory_;
    mutable std::mutex mutex_;
    
public:
    Value read(MemoryAddress addr, ThreadId tid);
    void write(MemoryAddress addr, Value value, ThreadId tid);
    Value atomicAdd(MemoryAddress addr, Value value, ThreadId tid);
    
    // Check for data races
    bool hasConflictingAccess(MemoryAddress addr, ThreadId tid1, ThreadId tid2) const;
    std::map<MemoryAddress, Value> getSnapshot() const;
    void clear();
};

// Barrier synchronization state
struct BarrierState {
    std::set<ThreadId> waitingThreads;
    std::set<ThreadId> arrivedThreads;
    std::mutex mutex;
    std::condition_variable cv;
    
    void reset(uint32_t totalThreads);
    bool tryArrive(ThreadId tid);
    void waitForAll();
};

// Threadgroup execution context
struct ThreadgroupContext {
    uint32_t threadgroupSize;
    uint32_t waveSize;
    uint32_t waveCount;
    std::vector<std::unique_ptr<WaveContext>> waves;
    std::shared_ptr<SharedMemory> sharedMemory;
    std::shared_ptr<BarrierState> currentBarrier;
    
    ThreadgroupContext(uint32_t tgSize, uint32_t wSize);
    ThreadId getGlobalThreadId(WaveId wid, LaneId lid) const;
    std::pair<WaveId, LaneId> getWaveAndLane(ThreadId tid) const;
};

// Thread execution ordering
struct ThreadOrdering {
    std::vector<ThreadId> executionOrder;
    std::string description;
    
    static ThreadOrdering sequential(uint32_t threadCount);
    static ThreadOrdering reverseSequential(uint32_t threadCount);
    static ThreadOrdering random(uint32_t threadCount, uint32_t seed);
    static ThreadOrdering evenOddInterleaved(uint32_t threadCount);
    static ThreadOrdering waveInterleaved(uint32_t threadCount, uint32_t waveSize);
};

// Execution result
struct ExecutionResult {
    std::map<std::string, Value> globalVariables;
    std::map<MemoryAddress, Value> sharedMemoryState;
    std::vector<Value> threadReturnValues;
    bool hasDataRace = false;
    std::string errorMessage;
    
    bool isValid() const { return errorMessage.empty() && !hasDataRace; }
};

// Expression AST nodes
class Expression {
public:
    virtual ~Expression() = default;
    virtual Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const = 0;
    virtual bool isDeterministic() const = 0;
    virtual std::string toString() const = 0;
};

// Pure expressions
class LiteralExpr : public Expression {
    Value value_;
public:
    explicit LiteralExpr(Value v) : value_(v) {}
    Value evaluate(LaneContext&, WaveContext&, ThreadgroupContext&) const override { return value_; }
    bool isDeterministic() const override { return true; }
    std::string toString() const override { return value_.toString(); }
};

class VariableExpr : public Expression {
    std::string name_;
public:
    explicit VariableExpr(const std::string& name) : name_(name) {}
    Value evaluate(LaneContext& lane, WaveContext&, ThreadgroupContext&) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override { return name_; }
};

class LaneIndexExpr : public Expression {
public:
    Value evaluate(LaneContext& lane, WaveContext&, ThreadgroupContext&) const override;
    bool isDeterministic() const override { return true; }
    std::string toString() const override { return "WaveGetLaneIndex()"; }
};

class WaveIndexExpr : public Expression {
public:
    Value evaluate(LaneContext&, WaveContext& wave, ThreadgroupContext&) const override;
    bool isDeterministic() const override { return true; }
    std::string toString() const override { return "WaveGetWaveIndex()"; }
};

class ThreadIndexExpr : public Expression {
public:
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return true; }
    std::string toString() const override { return "GetThreadIndex()"; }
};

class BinaryOpExpr : public Expression {
public:
    enum OpType { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Le, Gt, Ge, And, Or };
private:
    std::unique_ptr<Expression> left_;
    std::unique_ptr<Expression> right_;
    OpType op_;
public:
    BinaryOpExpr(std::unique_ptr<Expression> left, std::unique_ptr<Expression> right, OpType op);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override;
    std::string toString() const override;
};

class UnaryOpExpr : public Expression {
public:
    enum OpType { Neg, Not };
private:
    std::unique_ptr<Expression> expr_;
    OpType op_;
public:
    UnaryOpExpr(std::unique_ptr<Expression> expr, OpType op);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override;
    std::string toString() const override;
};

// Wave operations
class WaveActiveOp : public Expression {
public:
    enum OpType { Sum, Product, Min, Max, And, Or, Xor, CountBits };
private:
    std::unique_ptr<Expression> expr_;
    OpType op_;
public:
    WaveActiveOp(std::unique_ptr<Expression> expr, OpType op);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override;
};

class WaveGetLaneCountExpr : public Expression {
public:
    Value evaluate(LaneContext&, WaveContext& wave, ThreadgroupContext&) const override;
    bool isDeterministic() const override { return true; }
    std::string toString() const override { return "WaveGetLaneCount()"; }
};

// Statement AST nodes
class Statement {
public:
    virtual ~Statement() = default;
    virtual void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) = 0;
    virtual bool requiresAllLanesActive() const { return false; }
    virtual std::string toString() const = 0;
};

class VarDeclStmt : public Statement {
    std::string name_;
    std::unique_ptr<Expression> init_;
public:
    VarDeclStmt(const std::string& name, std::unique_ptr<Expression> init);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class AssignStmt : public Statement {
    std::string name_;
    std::unique_ptr<Expression> expr_;
public:
    AssignStmt(const std::string& name, std::unique_ptr<Expression> expr);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class IfStmt : public Statement {
    std::unique_ptr<Expression> condition_;
    std::vector<std::unique_ptr<Statement>> thenBlock_;
    std::vector<std::unique_ptr<Statement>> elseBlock_;
public:
    IfStmt(std::unique_ptr<Expression> cond, 
           std::vector<std::unique_ptr<Statement>> thenBlock,
           std::vector<std::unique_ptr<Statement>> elseBlock = {});
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    bool requiresAllLanesActive() const override;
    std::string toString() const override;
};

class ForStmt : public Statement {
    std::string loopVar_;
    std::unique_ptr<Expression> init_;
    std::unique_ptr<Expression> condition_;
    std::unique_ptr<Expression> increment_;
    std::vector<std::unique_ptr<Statement>> body_;
public:
    ForStmt(const std::string& var, 
            std::unique_ptr<Expression> init,
            std::unique_ptr<Expression> cond,
            std::unique_ptr<Expression> inc,
            std::vector<std::unique_ptr<Statement>> body);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class ReturnStmt : public Statement {
    std::unique_ptr<Expression> expr_;
public:
    explicit ReturnStmt(std::unique_ptr<Expression> expr = nullptr);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class BarrierStmt : public Statement {
public:
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    bool requiresAllLanesActive() const override { return true; }
    std::string toString() const override { return "GroupMemoryBarrierWithGroupSync();"; }
};

class SharedWriteStmt : public Statement {
    MemoryAddress addr_;
    std::unique_ptr<Expression> expr_;
public:
    SharedWriteStmt(MemoryAddress addr, std::unique_ptr<Expression> expr);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class SharedReadExpr : public Expression {
    MemoryAddress addr_;
public:
    explicit SharedReadExpr(MemoryAddress addr) : addr_(addr) {}
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override;
};

// Program representation
struct Program {
    std::vector<std::unique_ptr<Statement>> statements;
    uint32_t numThreadsX = 32;
    uint32_t numThreadsY = 1;
    uint32_t numThreadsZ = 1;
    
    uint32_t getTotalThreads() const { return numThreadsX * numThreadsY * numThreadsZ; }
};

// Main interpreter class
class MiniHLSLInterpreter {
private:
    static constexpr uint32_t DEFAULT_NUM_ORDERINGS = 10;
    std::mt19937 rng_;
    
    ExecutionResult executeWithOrdering(const Program& program, 
                                      const ThreadOrdering& ordering);
    
    void executeThread(ThreadId tid, 
                      const Program& program,
                      ThreadgroupContext& tgContext);
    
    static bool areResultsEquivalent(const ExecutionResult& r1, 
                                   const ExecutionResult& r2,
                                   double epsilon = 1e-6);
                                   
public:
    explicit MiniHLSLInterpreter(uint32_t seed = 42) : rng_(seed) {}
    
    // Execute with multiple random orderings and verify order independence
    struct VerificationResult {
        bool isOrderIndependent;
        std::vector<ExecutionResult> results;
        std::vector<ThreadOrdering> orderings;
        std::string divergenceReport;
    };
    
    VerificationResult verifyOrderIndependence(const Program& program, 
                                              uint32_t numOrderings = DEFAULT_NUM_ORDERINGS);
    
    // Execute with a specific ordering
    ExecutionResult execute(const Program& program, const ThreadOrdering& ordering);
    
    // Generate test orderings
    std::vector<ThreadOrdering> generateTestOrderings(uint32_t threadCount, 
                                                     uint32_t numOrderings);
};

// Helper functions for building programs
std::unique_ptr<Expression> makeLiteral(Value v);
std::unique_ptr<Expression> makeVariable(const std::string& name);
std::unique_ptr<Expression> makeLaneIndex();
std::unique_ptr<Expression> makeWaveIndex();
std::unique_ptr<Expression> makeThreadIndex();
std::unique_ptr<Expression> makeBinaryOp(std::unique_ptr<Expression> left,
                                        std::unique_ptr<Expression> right,
                                        BinaryOpExpr::OpType op);
std::unique_ptr<Expression> makeWaveSum(std::unique_ptr<Expression> expr);
std::unique_ptr<Statement> makeVarDecl(const std::string& name, 
                                      std::unique_ptr<Expression> init);
std::unique_ptr<Statement> makeAssign(const std::string& name,
                                     std::unique_ptr<Expression> expr);
std::unique_ptr<Statement> makeIf(std::unique_ptr<Expression> cond,
                                 std::vector<std::unique_ptr<Statement>> thenBlock,
                                 std::vector<std::unique_ptr<Statement>> elseBlock = {});

} // namespace interpreter
} // namespace minihlsl