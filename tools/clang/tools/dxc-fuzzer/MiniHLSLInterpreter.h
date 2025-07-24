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

// Forward declarations for Clang AST (to avoid heavy includes in header)
namespace clang {
    class FunctionDecl;
    class ASTContext;
    class Stmt;
    class Expr;
    class CompoundStmt;
    class BinaryOperator;
    class CallExpr;
    class DeclStmt;
    class CXXOperatorCallExpr;
    class IfStmt;
    class ForStmt;
    class WhileStmt;
    class DoStmt;
    class SwitchStmt;
    class CaseStmt;
    class DefaultStmt;
    class BreakStmt;
    class ContinueStmt;
    class ConditionalOperator;
    class FloatingLiteral;
    class CXXBoolLiteralExpr;
}

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

// Thread execution state for cooperative scheduling
enum class ThreadState {
    Ready,           // Ready to execute
    WaitingAtBarrier, // Waiting for barrier synchronization  
    WaitingForWave,  // Waiting for wave operation to complete
    Completed,       // Thread has finished execution
    Error            // Thread encountered an error
};

// Thread execution context
struct LaneContext {
    LaneId laneId;
    std::map<std::string, Value> variables;
    bool isActive = true;
    bool hasReturned = false;
    Value returnValue;
    
    // Execution state for cooperative scheduling
    ThreadState state = ThreadState::Ready;
    size_t currentStatement = 0;  // Index of next statement to execute
    std::string errorMessage;
    
    // Wave operation synchronization
    uint32_t waveOpId = 0;        // Current wave operation ID
    bool waveOpComplete = false;   // Whether current wave op is done
};

// Wave operation synchronization point for instruction-level coordination
struct WaveOperationSyncPoint {
    const void* instruction;              // Specific wave operation instruction pointer
    uint32_t blockId;                     // Which execution block this sync point belongs to
    std::set<LaneId> expectedParticipants; // Lanes that should participate (from block)
    std::set<LaneId> arrivedParticipants;  // Lanes that have arrived at THIS instruction
    std::map<LaneId, Value> pendingResults; // Results from arrived lanes
    bool allParticipantsKnown = false;    // All unknown lanes resolved for this block
    bool allParticipantsArrived = false;  // All expected participants arrived at instruction
    bool isComplete = false;              // Ready to execute
    
    // Instruction identification
    std::string instructionType;          // "WaveActiveSum", "WaveActiveAllTrue", etc.
    const void* sourceExpression = nullptr; // Source AST expression
};

// Legacy wave operation state (keeping for compatibility)
struct WaveOpState {
    uint32_t opId;
    WaveId waveId;
    std::set<LaneId> participatingLanes;  // Lanes that need to participate
    std::set<LaneId> completedLanes;      // Lanes that have completed the op
    std::map<LaneId, Value> results;      // Results from each lane
    bool isComplete = false;
};

// Barrier state for threadgroup synchronization  
struct ThreadgroupBarrierState {
    uint32_t barrierId;
    std::set<ThreadId> participatingThreads;  // All threads that must reach barrier
    std::set<ThreadId> arrivedThreads;        // Threads that have arrived
    bool isComplete = false;
};

// Instruction identity for tracking specific instructions within dynamic blocks
struct InstructionIdentity {
    const void* instruction = nullptr;       // Specific instruction pointer
    std::string instructionType;            // "WaveActiveSum", "WaveActiveAllTrue", etc.
    const void* sourceExpression = nullptr; // Source AST expression
    
    bool operator<(const InstructionIdentity& other) const {
        if (instruction != other.instruction) return instruction < other.instruction;
        if (instructionType != other.instructionType) return instructionType < other.instructionType;
        return sourceExpression < other.sourceExpression;
    }
    
    bool operator==(const InstructionIdentity& other) const {
        return instruction == other.instruction &&
               instructionType == other.instructionType &&
               sourceExpression == other.sourceExpression;
    }
};

// Merge stack entry for tracking control flow convergence points
struct MergeStackEntry {
    const void* sourceStatement = nullptr;  // Statement that created the divergence
    uint32_t parentBlockId = 0;             // Block before divergence
    std::set<uint32_t> divergentBlockIds;   // Blocks that will converge
    
    bool operator<(const MergeStackEntry& other) const {
        if (sourceStatement != other.sourceStatement) return sourceStatement < other.sourceStatement;
        if (parentBlockId != other.parentBlockId) return parentBlockId < other.parentBlockId;
        return divergentBlockIds < other.divergentBlockIds;
    }
    
    bool operator==(const MergeStackEntry& other) const {
        return sourceStatement == other.sourceStatement &&
               parentBlockId == other.parentBlockId &&
               divergentBlockIds == other.divergentBlockIds;
    }
};

// Block identity for deduplication based on execution path using merge stack
struct BlockIdentity {
    const void* sourceStatement = nullptr;  // Which statement created this block
    bool conditionValue = true;             // Which branch (true/false)
    uint32_t parentBlockId = 0;             // Parent block for nested control flow
    std::vector<MergeStackEntry> mergeStack; // Stack of merge points for robust identification
    
    bool operator<(const BlockIdentity& other) const {
        if (sourceStatement != other.sourceStatement) return sourceStatement < other.sourceStatement;
        if (conditionValue != other.conditionValue) return conditionValue < other.conditionValue;
        if (parentBlockId != other.parentBlockId) return parentBlockId < other.parentBlockId;
        return mergeStack < other.mergeStack;
    }
    
    bool operator==(const BlockIdentity& other) const {
        return sourceStatement == other.sourceStatement &&
               conditionValue == other.conditionValue &&
               parentBlockId == other.parentBlockId &&
               mergeStack == other.mergeStack;
    }
};

// Dynamic execution block for SIMT control flow
struct DynamicExecutionBlock {
    uint32_t blockId;
    BlockIdentity identity;               // Unique identity for this execution path
    std::set<LaneId> participatingLanes;  // Lanes currently executing in this block
    uint32_t programPoint;                // Current execution point within the block
    uint32_t parentBlockId = 0;           // Parent block for nested control flow
    bool isConverged = true;              // Whether all lanes in wave are in this block
    
    // Control flow context
    const void* sourceStatement = nullptr; // Source AST statement for this block
    int nestingLevel = 0;                 // Nesting depth for control flow
    
    // Instruction tracking for synchronized operations
    std::vector<InstructionIdentity> instructionList; // Ordered list of instructions in this block
    std::map<InstructionIdentity, std::set<LaneId>> instructionParticipants; // Which lanes participate in each instruction
    
    // Cooperative scheduling state
    std::set<LaneId> unknownLanes;        // Lanes that haven't reached this control flow point yet
    std::set<LaneId> arrivedLanes;        // Lanes that have arrived at this block
    std::set<LaneId> waitingLanes;        // Lanes waiting for wave operations in this block
    bool allUnknownResolved = false;      // Whether all unknown lanes are resolved
};

// Wave execution context
struct WaveContext {
    WaveId waveId;
    uint32_t waveSize;
    std::vector<std::unique_ptr<LaneContext>> lanes;
    
    // Wave operation synchronization
    std::map<uint32_t, WaveOpState> activeWaveOps;
    uint32_t nextWaveOpId = 1;
    
    // Instruction-level synchronization
    std::map<const void*, WaveOperationSyncPoint> activeSyncPoints; // instruction -> sync point
    std::map<LaneId, const void*> laneWaitingAtInstruction; // which instruction each lane is waiting at
    
    // Dynamic execution block tracking
    std::map<uint32_t, DynamicExecutionBlock> executionBlocks;
    std::map<BlockIdentity, uint32_t> identityToBlockId; // Deduplication map
    uint32_t nextBlockId = 1;
    std::map<LaneId, uint32_t> laneToCurrentBlock; // Which block each lane is in
    
    // Merge stack for tracking control flow convergence
    std::vector<MergeStackEntry> globalMergeStack; // Global merge stack for all lanes
    std::map<LaneId, std::vector<MergeStackEntry>> laneMergeStacks; // Per-lane merge stacks
    
    // Get active lane mask (based on current control flow)
    uint64_t getActiveMask() const;
    std::vector<LaneId> getActiveLanes() const;
    std::vector<LaneId> getCurrentlyActiveLanes() const; // Only lanes with isActive=true
    bool allLanesActive() const;
    uint32_t countActiveLanes() const;
    uint32_t countCurrentlyActiveLanes() const;
    
    // Dynamic execution block methods
    uint32_t createExecutionBlock(const std::set<LaneId>& lanes, const void* sourceStmt = nullptr);
    void assignLaneToBlock(LaneId laneId, uint32_t blockId);
    std::vector<LaneId> getLanesInSameBlock(LaneId laneId) const;
    uint32_t getCurrentBlock(LaneId laneId) const;
    bool areInSameBlock(LaneId lane1, LaneId lane2) const;
    void mergeExecutionPaths(const std::vector<uint32_t>& blockIds, uint32_t targetBlockId);
    
    // Cooperative scheduling methods
    void markLaneArrived(LaneId laneId, uint32_t blockId);
    void markLaneWaitingForWave(LaneId laneId, uint32_t blockId);
    bool canExecuteWaveOperation(LaneId laneId) const;
    std::vector<LaneId> getWaveOperationParticipants(LaneId laneId) const;
    void resolveUnknownLane(LaneId laneId, uint32_t chosenBlockId);
    bool areAllUnknownLanesResolved(uint32_t blockId) const;
    
    // Block deduplication methods
    uint32_t findOrCreateBlockForPath(const BlockIdentity& identity, const std::set<LaneId>& unknownLanes);
    uint32_t findBlockByIdentity(const BlockIdentity& identity) const;
    BlockIdentity createBlockIdentity(const void* sourceStmt, bool conditionValue, 
                                     uint32_t parentBlockId, const std::vector<MergeStackEntry>& mergeStack = {}) const;
    
    // Proactive block creation for control flow
    std::pair<uint32_t, uint32_t> createIfBlocks(const void* ifStmt, uint32_t parentBlockId, 
                                                  const std::vector<MergeStackEntry>& mergeStack, bool hasElse);
    uint32_t createLoopIterationBlock(const void* loopStmt, uint32_t parentBlockId, 
                                      const std::vector<MergeStackEntry>& mergeStack);
    std::vector<uint32_t> createSwitchCaseBlocks(const void* switchStmt, uint32_t parentBlockId,
                                                  const std::vector<MergeStackEntry>& mergeStack,
                                                  const std::vector<int>& caseValues, bool hasDefault);
    void moveThreadFromUnknownToParticipating(uint32_t blockId, LaneId laneId);
    void removeThreadFromUnknown(uint32_t blockId, LaneId laneId);
    void removeThreadFromNestedBlocks(uint32_t parentBlockId, LaneId laneId);
    std::set<LaneId> getCurrentBlockParticipants(uint32_t blockId) const;
    
    // Merge stack management
    void pushMergePoint(LaneId laneId, const void* sourceStmt, uint32_t parentBlockId, const std::set<uint32_t>& divergentBlocks);
    void popMergePoint(LaneId laneId);
    std::vector<MergeStackEntry> getCurrentMergeStack(LaneId laneId) const;
    void updateMergeStack(LaneId laneId, const std::vector<MergeStackEntry>& mergeStack);
    
    // Instruction identity management
    void addInstructionToBlock(uint32_t blockId, const InstructionIdentity& instruction, const std::set<LaneId>& participants);
    InstructionIdentity createInstructionIdentity(const void* instruction, const std::string& instructionType, const void* sourceExpr = nullptr) const;
    std::vector<InstructionIdentity> getBlockInstructions(uint32_t blockId) const;
    std::set<LaneId> getInstructionParticipantsInBlock(uint32_t blockId, const InstructionIdentity& instruction) const;
    bool canExecuteInstructionInBlock(uint32_t blockId, const InstructionIdentity& instruction) const;
    std::set<LaneId> getExpectedParticipantsInBlock(uint32_t blockId, const InstructionIdentity& instruction) const;
    
    // Instruction-level synchronization methods
    bool canExecuteWaveInstruction(LaneId laneId, const void* instruction) const;
    void markLaneArrivedAtInstruction(LaneId laneId, const void* instruction, const std::string& instructionType);
    bool areAllParticipantsKnownForInstruction(const void* instruction) const;
    bool haveAllParticipantsArrivedAtInstruction(const void* instruction) const;
    std::vector<LaneId> getInstructionParticipants(const void* instruction) const;
    void createOrUpdateSyncPoint(const void* instruction, LaneId laneId, const std::string& instructionType);
    void releaseSyncPoint(const void* instruction);
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
    
    // Barrier synchronization
    std::map<uint32_t, ThreadgroupBarrierState> activeBarriers;
    uint32_t nextBarrierId = 1;
    
    ThreadgroupContext(uint32_t tgSize, uint32_t wSize);
    ThreadId getGlobalThreadId(WaveId wid, LaneId lid) const;
    std::pair<WaveId, LaneId> getWaveAndLane(ThreadId tid) const;
    
    // Cooperative scheduling helpers
    std::vector<ThreadId> getReadyThreads() const;
    std::vector<ThreadId> getWaitingThreads() const;
    bool canExecuteWaveOp(WaveId waveId, const std::set<LaneId>& activeLanes) const;
    bool canReleaseBarrier(uint32_t barrierId) const;
    
    // Global dynamic block creation methods (delegates to appropriate wave)
    std::pair<uint32_t, uint32_t> createIfBlocks(const void* ifStmt, uint32_t parentBlockId, 
                                                  const std::vector<MergeStackEntry>& mergeStack, bool hasElse, WaveId waveId);
    uint32_t createLoopIterationBlock(const void* loopStmt, uint32_t parentBlockId, 
                                      const std::vector<MergeStackEntry>& mergeStack, WaveId waveId);
    std::vector<uint32_t> createSwitchCaseBlocks(const void* switchStmt, uint32_t parentBlockId,
                                                  const std::vector<MergeStackEntry>& mergeStack,
                                                  const std::vector<int>& caseValues, bool hasDefault, WaveId waveId);
    void moveThreadFromUnknownToParticipating(uint32_t blockId, LaneId laneId, WaveId waveId);
    void removeThreadFromUnknown(uint32_t blockId, LaneId laneId, WaveId waveId);
    void removeThreadFromNestedBlocks(uint32_t parentBlockId, LaneId laneId, WaveId waveId);
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
    std::string toString() const override { return "W()"; }
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

class ConditionalExpr : public Expression {
private:
    std::unique_ptr<Expression> condition_;
    std::unique_ptr<Expression> trueExpr_;
    std::unique_ptr<Expression> falseExpr_;
public:
    ConditionalExpr(std::unique_ptr<Expression> condition, 
                   std::unique_ptr<Expression> trueExpr, 
                   std::unique_ptr<Expression> falseExpr);
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

class WaveIsFirstLaneExpr : public Expression {
public:
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override { return "WaveIsFirstLane()"; }
};

class WaveActiveAllEqualExpr : public Expression {
private:
    std::unique_ptr<Expression> expr_;
public:
    explicit WaveActiveAllEqualExpr(std::unique_ptr<Expression> expr);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override;
};

class WaveActiveAllTrueExpr : public Expression {
private:
    std::unique_ptr<Expression> expr_;
public:
    explicit WaveActiveAllTrueExpr(std::unique_ptr<Expression> expr);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override;
};

class WaveActiveAnyTrueExpr : public Expression {
private:
    std::unique_ptr<Expression> expr_;
public:
    explicit WaveActiveAnyTrueExpr(std::unique_ptr<Expression> expr);
    Value evaluate(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) const override;
    bool isDeterministic() const override { return false; }
    std::string toString() const override;
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

class WhileStmt : public Statement {
    std::unique_ptr<Expression> condition_;
    std::vector<std::unique_ptr<Statement>> body_;
public:
    WhileStmt(std::unique_ptr<Expression> cond,
              std::vector<std::unique_ptr<Statement>> body);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class DoWhileStmt : public Statement {
    std::vector<std::unique_ptr<Statement>> body_;
    std::unique_ptr<Expression> condition_;
public:
    DoWhileStmt(std::vector<std::unique_ptr<Statement>> body,
                std::unique_ptr<Expression> cond);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

class SwitchStmt : public Statement {
    std::unique_ptr<Expression> condition_;
    struct CaseBlock {
        std::optional<int> value; // nullopt for default case
        std::vector<std::unique_ptr<Statement>> statements;
    };
    std::vector<CaseBlock> cases_;
public:
    SwitchStmt(std::unique_ptr<Expression> cond);
    void addCase(int value, std::vector<std::unique_ptr<Statement>> stmts);
    void addDefault(std::vector<std::unique_ptr<Statement>> stmts);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
};

// Control flow exceptions for break/continue
class ControlFlowException : public std::exception {
public:
    enum Type { Break, Continue };
    Type type;
    explicit ControlFlowException(Type t) : type(t) {}
};

class BreakStmt : public Statement {
public:
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override { return "break;"; }
};

class ContinueStmt : public Statement {
public:
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override { return "continue;"; }
};

class ReturnStmt : public Statement {
    std::unique_ptr<Expression> expr_;
public:
    explicit ReturnStmt(std::unique_ptr<Expression> expr = nullptr);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
    
private:
    void handleGlobalEarlyReturn(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg);
    void updateBlockResolutionStates(WaveContext& wave, LaneId returningLaneId);
    void updateWaveOperationStates(WaveContext& wave, LaneId returningLaneId);
    void updateBarrierStates(ThreadgroupContext& tg, LaneId returningLaneId);
};

class BarrierStmt : public Statement {
public:
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    bool requiresAllLanesActive() const override { return true; }
    std::string toString() const override { return "GroupMemoryBarrierWithGroupSync();"; }
};

class ExprStmt : public Statement {
    std::unique_ptr<Expression> expr_;
public:
    explicit ExprStmt(std::unique_ptr<Expression> expr);
    void execute(LaneContext& lane, WaveContext& wave, ThreadgroupContext& tg) override;
    std::string toString() const override;
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
    
    // Cooperative execution engine
    bool executeOneStep(ThreadId tid, const Program& program, ThreadgroupContext& tgContext);
    void processWaveOperations(ThreadgroupContext& tgContext);
    void processBarriers(ThreadgroupContext& tgContext);
    ThreadId selectNextThread(const std::vector<ThreadId>& readyThreads, 
                             const ThreadOrdering& ordering, 
                             uint32_t& orderingIndex);
    
    static bool areResultsEquivalent(const ExecutionResult& r1, 
                                   const ExecutionResult& r2,
                                   double epsilon = 1e-6);
    
    // HLSL AST conversion helper methods (simplified for now)
    std::unique_ptr<Statement> convertStatement(const clang::Stmt* stmt, clang::ASTContext& context);
    std::unique_ptr<Expression> convertExpression(const clang::Expr* expr, clang::ASTContext& context);
                                   
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
    
    // HLSL AST conversion methods
    struct ConversionResult {
        bool success;
        std::string errorMessage;
        Program program;
    };
    
    // Convert Clang AST function to interpreter program
    ConversionResult convertFromHLSLAST(const clang::FunctionDecl* func, clang::ASTContext& context);

private:
    // AST conversion helper methods (already declared above: convertStatement, convertExpression)
    void extractThreadConfiguration(const clang::FunctionDecl* func, Program& program);
    void convertCompoundStatement(const clang::CompoundStmt* compound, Program& program, clang::ASTContext& context);
    std::unique_ptr<Statement> convertBinaryOperator(const clang::BinaryOperator* binOp, clang::ASTContext& context);
    std::unique_ptr<Statement> convertCallExpression(const clang::CallExpr* callExpr, clang::ASTContext& context);
    std::unique_ptr<Statement> convertDeclarationStatement(const clang::DeclStmt* declStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertIfStatement(const clang::IfStmt* ifStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertForStatement(const clang::ForStmt* forStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertWhileStatement(const clang::WhileStmt* whileStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertDoStatement(const clang::DoStmt* doStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertSwitchStatement(const clang::SwitchStmt* switchStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertBreakStatement(const clang::BreakStmt* breakStmt, clang::ASTContext& context);
    std::unique_ptr<Statement> convertContinueStatement(const clang::ContinueStmt* continueStmt, clang::ASTContext& context);
    std::unique_ptr<Expression> convertCallExpressionToExpression(const clang::CallExpr* callExpr, clang::ASTContext& context);
    std::unique_ptr<Expression> convertConditionalOperator(const clang::ConditionalOperator* condOp, clang::ASTContext& context);
    std::unique_ptr<Expression> convertBinaryExpression(const clang::BinaryOperator* binOp, clang::ASTContext& context);
    std::unique_ptr<Expression> convertOperatorCall(const clang::CXXOperatorCallExpr* opCall, clang::ASTContext& context);
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