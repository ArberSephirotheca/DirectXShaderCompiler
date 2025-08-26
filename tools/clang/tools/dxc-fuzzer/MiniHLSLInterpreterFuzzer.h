#pragma once

#include "MiniHLSLInterpreter.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <set>

// Forward declarations for interpreter types
namespace minihlsl {
namespace interpreter {
class Statement;
class Expression;
class IfStmt;
class ForStmt;
class WhileStmt;
class DoWhileStmt;
struct Program;
class LaneIndexExpr;
class WaveIndexExpr;
class BinaryOpExpr;
class UnaryOpExpr;
class LiteralExpr;
class AssignStmt;
class WaveActiveOp;
class Interpreter;
} // namespace interpreter

namespace fuzzer {

// Forward declarations
struct ExecutionTrace;
class MutationStrategy;
class SemanticValidator;
struct GenerationRound;

// ===== Execution Trace Data Structures =====

struct ExecutionTrace {
  // Thread hierarchy information
  struct ThreadHierarchy {
    uint32_t totalThreads;
    uint32_t waveSize;
    uint32_t numWaves;
    
    std::map<interpreter::ThreadId, interpreter::WaveId> threadToWave;
    std::map<interpreter::ThreadId, interpreter::LaneId> threadToLane;
    std::map<interpreter::WaveId, std::vector<interpreter::ThreadId>> waveToThreads;
  };
  ThreadHierarchy threadHierarchy;
  
  // Dynamic block execution graph
  struct BlockExecutionRecord {
    uint32_t blockId;
    interpreter::BlockIdentity identity;
    interpreter::BlockType blockType;
    
    // Creation context
    const void* sourceStatement;
    uint32_t parentBlockId;
    interpreter::WaveId creatorWave;
    
    // Participation tracking (matches BlockMembershipRegistry)
    struct WaveParticipation {
      std::set<interpreter::LaneId> participatingLanes;
      std::set<interpreter::LaneId> arrivedLanes;
      std::set<interpreter::LaneId> unknownLanes;
      std::set<interpreter::LaneId> waitingForWaveLanes;
      
      // Visit count per lane
      std::map<interpreter::LaneId, uint32_t> visitCount;
      
      // Entry/exit order for each lane
      std::map<interpreter::LaneId, std::vector<uint64_t>> entryTimestamps;
      std::map<interpreter::LaneId, std::vector<uint64_t>> exitTimestamps;
    };
    std::map<interpreter::WaveId, WaveParticipation> waveParticipation;
    
    // Block relationships
    std::set<uint32_t> predecessors;
    std::set<uint32_t> successors;
    
    // Merge information
    bool isMergePoint;
    std::set<uint32_t> mergedFromBlocks;
    
    // Loop iteration information (if this is a loop body block)
    struct LoopIterationInfo {
      std::string loopVariable;  // e.g., "i", "counter0"
      int iterationValue;        // Current value: 0, 1, 2...
      uint32_t loopHeaderBlock;  // The LOOP_HEADER block ID
    };
    std::optional<LoopIterationInfo> loopIteration;
  };
  std::map<uint32_t, BlockExecutionRecord> blocks;
  
  // Control flow decisions
  struct ControlFlowDecision {
    const void* statement;
    uint32_t executionBlockId;
    uint64_t timestamp;
    
    struct LaneDecision {
      interpreter::Value conditionValue;
      bool branchTaken;
      interpreter::LaneContext::ControlFlowPhase phase;
      uint32_t nextBlockId;
    };
    
    // Per-wave, per-lane decisions
    std::map<interpreter::WaveId, std::map<interpreter::LaneId, LaneDecision>> decisions;
  };
  std::vector<ControlFlowDecision> controlFlowHistory;
  
  // Wave operation synchronization
  struct WaveOpRecord {
    // Identification
    uint32_t stableId = 0;  // Stable ID of the wave operation expression
    const void* instruction;
    std::string opType;
    uint32_t blockId;
    interpreter::WaveId waveId;
    uint64_t syncPointId;
    int waveOpEnumType = -1; // WaveActiveOp::OpType enum value
    
    // Participants and values
    std::set<interpreter::LaneId> expectedParticipants;
    std::set<interpreter::LaneId> arrivedParticipants;
    std::map<interpreter::LaneId, interpreter::Value> inputValues;
    std::map<interpreter::LaneId, interpreter::Value> outputValues;
    
    // Synchronization behavior
    struct SyncBehavior {
      std::vector<interpreter::LaneId> arrivalOrder;
      std::map<interpreter::LaneId, uint64_t> arrivalTime;
      std::map<interpreter::LaneId, uint64_t> resumeTime;
      bool anyLaneWaited;
      uint64_t totalWaitTime;
    };
    SyncBehavior syncBehavior;
    
    // Final state from sync point
    interpreter::SyncPointState finalState;
  };
  std::vector<WaveOpRecord> waveOperations;
  
  // Loop execution patterns
  struct LoopExecutionPattern {
    const void* loopStatement;
    
    struct LanePattern {
      uint32_t totalIterations;
      std::vector<uint32_t> iterationBlocks;
      bool exitedEarly;
      uint32_t breakIteration;
      
      // Condition values per iteration
      std::vector<interpreter::Value> conditionHistory;
    };
    
    // Per-wave, per-lane patterns
    std::map<interpreter::WaveId, std::map<interpreter::LaneId, LanePattern>> lanePatterns;
    
    // Loop body variations
    std::set<std::vector<uint32_t>> uniqueBodyPaths;
  };
  std::map<const void*, LoopExecutionPattern> loops;
  
  // Variable access tracking
  struct VariableAccess {
    std::string varName;
    const void* accessSite;
    uint32_t blockId;
    bool isWrite;
    
    // Values per wave/lane at access time
    std::map<interpreter::WaveId, std::map<interpreter::LaneId, interpreter::Value>> values;
    
    // For detecting data races
    uint64_t timestamp;
    enum AccessType { Read, Write, ReadModifyWrite };
    AccessType type;
  };
  std::vector<VariableAccess> variableAccesses;
  
  // Memory access patterns
  struct MemoryAccess {
    enum MemoryType { SharedMemory, GlobalBuffer };
    MemoryType type;
    std::string bufferName;
    uint32_t address;
    
    interpreter::ThreadId accessingThread;
    interpreter::WaveId waveId;
    interpreter::LaneId laneId;
    
    bool isAtomic;
    std::string atomicOp;
    interpreter::Value oldValue;
    interpreter::Value newValue;
    
    uint64_t timestamp;
    uint32_t blockId;
  };
  std::vector<MemoryAccess> memoryAccesses;
  
  // Threadgroup barriers
  struct BarrierRecord {
    uint32_t barrierId;
    const void* barrierSite;
    
    // Arrival pattern
    std::map<interpreter::ThreadId, uint64_t> arrivalTimes;
    std::map<interpreter::ThreadId, uint64_t> releaseTimes;
    
    // Wave synchronization
    std::map<interpreter::WaveId, std::set<interpreter::LaneId>> arrivedLanesPerWave;
    std::vector<interpreter::WaveId> waveArrivalOrder;
    
    uint64_t totalWaitTime;
  };
  std::vector<BarrierRecord> barriers;
  
  // Final program state
  struct FinalState {
    // Variable values per lane
    std::map<interpreter::WaveId, 
             std::map<interpreter::LaneId, 
                      std::map<std::string, interpreter::Value>>> laneVariables;
    
    // Return values
    std::map<interpreter::WaveId, 
             std::map<interpreter::LaneId, interpreter::Value>> returnValues;
    
    // Memory state
    std::map<std::string, std::map<uint32_t, interpreter::Value>> globalBuffers;
    std::map<interpreter::MemoryAddress, interpreter::Value> sharedMemory;
    
    // Thread states
    std::map<interpreter::WaveId, 
             std::map<interpreter::LaneId, interpreter::ThreadState>> finalThreadStates;
  };
  FinalState finalState;
  
  // Execution statistics
  struct Statistics {
    uint64_t totalInstructions;
    uint64_t totalWaveOps;
    uint64_t totalBarriers;
    uint32_t maxDivergenceDepth;
    uint32_t totalDynamicBlocks;
    uint64_t totalSyncWaitTime;
  };
  Statistics stats;
  
  // Store final execution result data
  std::map<interpreter::MemoryAddress, interpreter::Value> finalMemoryState;
  std::map<std::string, interpreter::Value> finalVariableStates;
};

// ===== Trace Capture Interpreter =====

// TraceCaptureInterpreter - defined in separate files
class TraceCaptureInterpreter;

// ===== Mutation Strategies =====

class MutationStrategy {
public:
  virtual ~MutationStrategy() = default;
  
  // Check if this mutation can be applied given the trace
  virtual bool canApply(const interpreter::Statement* stmt, 
                       const ExecutionTrace& trace) const = 0;
  
  // Apply the mutation
  virtual std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const = 0;
  
  // Verify the mutation preserves semantics
  virtual bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const = 0;
  
  virtual std::string getName() const = 0;
  
  // New: Program-level mutation API
  virtual bool requiresProgramLevelMutation() const { return false; }
  
  virtual std::vector<interpreter::Program> applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
    // Default implementation returns empty vector
    return std::vector<interpreter::Program>();
  }
};

// ===== Semantics-Preserving Mutation Strategies =====

// Mutation strategy: Permute lane IDs in associative wave operations
class LanePermutationMutation : public MutationStrategy {
public:
  bool canApply(const interpreter::Statement* stmt, 
                const ExecutionTrace& trace) const override;
  
  std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const override;
  
  bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const override;
  
  std::string getName() const override { return "LanePermutation"; }
  
  // Program-level mutation API
  bool requiresProgramLevelMutation() const override { return true; }
  
  std::vector<interpreter::Program> applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const override;
  
private:
  // Permutation strategies
  enum class PermutationType {
    Rotate,      // (laneId + offset) % count
    Reverse,     // count - 1 - laneId
    EvenOddSwap, // laneId ^ 1
    BitReverse   // Reverse bits of laneId
  };
  
  // Helper to create permutation expression
  std::unique_ptr<interpreter::Expression> createPermutationExpr(
      PermutationType type,
      std::unique_ptr<interpreter::Expression> laneExpr,
      uint32_t activeLaneCount,
      const std::set<interpreter::LaneId>& participatingLanes) const;
  
  // Recursively transform expressions, replacing LaneIndexExpr
  std::unique_ptr<interpreter::Expression> transformExpression(
      const interpreter::Expression* expr,
      const interpreter::Expression* permutedLaneExpr) const;
  
  // Extract wave operation from ExprStmt (if any)
  const interpreter::WaveActiveOp* getWaveOp(const interpreter::Statement* stmt) const;
  
  // Check if expression uses built-in thread ID variables
  bool usesThreadIdVariables(const interpreter::Expression* expr) const;
  
  // Process statements recursively to apply lane permutations
  void processStatementsForPermutation(
      const std::vector<std::unique_ptr<interpreter::Statement>>& input,
      std::vector<std::unique_ptr<interpreter::Statement>>& output,
      const ExecutionTrace& trace,
      const std::set<size_t>& statementsToMutate,
      size_t& currentStmtIndex,
      bool& anyMutationApplied) const;
  
  // Apply permutation to a single statement if it contains a wave op
  bool applyPermutationToStatement(
      const interpreter::Statement* stmt,
      std::vector<std::unique_ptr<interpreter::Statement>>& output,
      const ExecutionTrace& trace) const;
  
  // Generate unique variable name for permutation
  std::string generatePermVarName() const;
  
  // Replace wave operation in expression tree
  std::unique_ptr<interpreter::Expression> replaceWaveOpInExpression(
      const interpreter::Expression* expr,
      const interpreter::WaveActiveOp* targetWaveOp,
      std::unique_ptr<interpreter::Expression> replacement) const;
};


// Mutation strategy: Track wave operation participants in global buffer
class WaveParticipantTrackingMutation : public MutationStrategy {
public:
  bool canApply(const interpreter::Statement* stmt,
                const ExecutionTrace& trace) const override;
  
  std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const override;
    
  bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const override;
    
  std::string getName() const override { return "WaveParticipantTracking"; }
  
  // Program-level mutation API
  bool requiresProgramLevelMutation() const override { return true; }
  
  std::vector<interpreter::Program> applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const override;
  
  // Process statements recursively to inject tracking after wave operations
  void processStatementsForTracking(
      const std::vector<std::unique_ptr<interpreter::Statement>>& input,
      std::vector<std::unique_ptr<interpreter::Statement>>& output,
      const ExecutionTrace& trace,
      size_t& nextTraceIndex) const;
  
private:
  // Helper to create participant tracking code
  std::vector<std::unique_ptr<interpreter::Statement>> 
  createTrackingStatements(const interpreter::WaveActiveOp* waveOp,
                          const std::string& resultVar,
                          uint32_t expectedParticipants) const;
  
  // Check if program already has a participant buffer
  bool hasParticipantBuffer(const interpreter::Program& program) const;
  
  // Helper methods for program-level mutation
  bool hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const;
    
  interpreter::Program createMutantWithTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const;
    
  void ensureParticipantBuffer(interpreter::Program& mutant) const;
  
  interpreter::Program createBaseMutant(const interpreter::Program& program) const;
};

// Mutation strategy: Track wave operation participation frequency per thread
class WaveParticipantFrequencyMutation : public MutationStrategy {
public:
  bool canApply(const interpreter::Statement* stmt,
                const ExecutionTrace& trace) const override;
  
  std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const override;
    
  bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const override;
    
  std::string getName() const override { return "WaveParticipantFrequency"; }
  
  // Program-level mutation API
  bool requiresProgramLevelMutation() const override { return true; }
  
  std::vector<interpreter::Program> applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const override;
  
private:
  // Process statements recursively to inject frequency tracking
  void processStatementsForFrequency(
      const std::vector<std::unique_ptr<interpreter::Statement>>& input,
      std::vector<std::unique_ptr<interpreter::Statement>>& output,
      const ExecutionTrace& trace,
      size_t& currentWaveOpIndex) const;
  
  // Helper to create frequency tracking code
  std::vector<std::unique_ptr<interpreter::Statement>> 
  createFrequencyTrackingStatements(const interpreter::WaveActiveOp* waveOp,
                                   const std::string& resultVar) const;
  
  // Check if program already has a frequency buffer
  bool hasFrequencyBuffer(const interpreter::Program& program) const;
  
  // Helper methods for program-level mutation
  bool hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const;
    
  interpreter::Program createMutantWithFrequencyTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const;
    
  void ensureFrequencyBuffer(interpreter::Program& mutant) const;
  
  interpreter::Program createBaseMutant(const interpreter::Program& program) const;
  
  // Compute expected frequencies from trace
  std::map<interpreter::ThreadId, uint32_t> computeExpectedFrequencies(
    const ExecutionTrace& trace) const;
};

// Mutation strategy: Track wave operation participants using bit masks
class WaveParticipantBitTrackingMutation : public MutationStrategy {
public:
  bool canApply(const interpreter::Statement* stmt,
                const ExecutionTrace& trace) const override;
  
  std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const override;
    
  bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const override;
    
  std::string getName() const override { return "WaveParticipantBitTracking"; }
  
  // Program-level mutation API
  bool requiresProgramLevelMutation() const override { return true; }
  
  std::vector<interpreter::Program> applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const override;
  
private:
  // Process statements recursively to inject bit tracking after wave operations
  void processStatementsForBitTracking(
      const std::vector<std::unique_ptr<interpreter::Statement>>& input,
      std::vector<std::unique_ptr<interpreter::Statement>>& output,
      const ExecutionTrace& trace,
      size_t& nextWaveOpIndex,
      std::map<const void*, uint32_t>& loopIterationMap,
      uint32_t currentLoopIteration,
      const std::string& currentLoopVar = "") const;
  
  // Helper to create bit tracking code
  std::vector<std::unique_ptr<interpreter::Statement>> 
  createBitTrackingStatements(const interpreter::WaveActiveOp* waveOp,
                             const std::string& resultVar,
                             uint32_t waveOpId,
                             uint32_t loopIteration,
                             const std::string& loopVarName = "") const;
  
  // Check if program already has bit tracking buffers
  bool hasBitTrackingBuffers(const interpreter::Program& program) const;
  
  // Helper methods for program-level mutation
  bool hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const;
    
  interpreter::Program createMutantWithBitTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const;
    
  void ensureBitTrackingBuffers(interpreter::Program& mutant) const;
  
  interpreter::Program createBaseMutant(const interpreter::Program& program) const;
  
  // Compute expected bit patterns from trace
  std::vector<uint32_t> computeExpectedBitPatterns(
    const ExecutionTrace& trace) const;
    
  // Get number of uints needed for wave mask (2 for wave32, 3 for wave64)
  uint32_t getRecordSize(uint32_t waveSize) const;
  
  // Assign stable IDs to wave operations
  void assignWaveOpIds(
    const std::vector<std::unique_ptr<interpreter::Statement>>& statements,
    std::map<const interpreter::WaveActiveOp*, uint32_t>& waveOpToId,
    uint32_t& nextId) const;
};

// Mutation strategy: Add participants to specific loop iterations
class ContextAwareParticipantMutation : public MutationStrategy {
public:
  bool canApply(const interpreter::Statement* stmt, 
                const ExecutionTrace& trace) const override;
  
  std::unique_ptr<interpreter::Statement> apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const override;
  
  bool validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const override;
  
  std::string getName() const override { return "ContextAwareParticipant"; }
  
private:
  // Extract loop iteration info from block's execution path
  struct IterationContext {
    std::string loopVariable;
    int iterationValue;
    uint32_t blockId;
    std::set<interpreter::LaneId> existingParticipants;
    interpreter::WaveId waveId;
  };
  
  std::vector<IterationContext> extractIterationContexts(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const;
  
  std::unique_ptr<interpreter::Expression> createIterationSpecificCondition(
    const IterationContext& context,
    interpreter::LaneId newLane) const;
    
  // Helper to find loop variable from parent blocks
  std::string findLoopVariable(
    uint32_t blockId,
    const ExecutionTrace& trace) const;
};


// ===== Semantic Validator =====

class SemanticValidator {
public:
  struct ValidationResult {
    bool isEquivalent;
    std::string divergenceReason;
    std::vector<std::string> differences;
  };
  
  ValidationResult validate(const ExecutionTrace& golden, 
                          const ExecutionTrace& mutant);
  
private:
  bool compareFinalStates(const ExecutionTrace::FinalState& golden,
                         const ExecutionTrace::FinalState& mutant,
                         ValidationResult& result);
  
  bool compareWaveOperations(const std::vector<ExecutionTrace::WaveOpRecord>& golden,
                            const std::vector<ExecutionTrace::WaveOpRecord>& mutant,
                            ValidationResult& result);
  
  bool compareMemoryState(const ExecutionTrace& golden,
                         const ExecutionTrace& mutant,
                         ValidationResult& result);
  
  bool verifyControlFlowEquivalence(const ExecutionTrace& golden,
                                   const ExecutionTrace& mutant,
                                   ValidationResult& result);
};

// ===== Bug Reporting =====

class BugReporter {
public:
  struct BugReport {
    std::string id;
    std::chrono::system_clock::time_point timestamp;
    
    // Program information
    std::string originalProgram;
    std::string mutantProgram;
    std::string mutation;
    
    // Trace comparison
    struct TraceDivergence {
      enum DivergenceType {
        BlockStructure,
        WaveOperation,
        ControlFlow,
        Synchronization,
        Memory
      };
      
      DivergenceType type;
      uint64_t divergencePoint;
      std::string description;
    };
    TraceDivergence traceDivergence;
    
    SemanticValidator::ValidationResult validation;
    
    // Classification
    enum BugType {
      WaveOpInconsistency,
      ReconvergenceError,
      DeadlockOrRace,
      MemoryCorruption,
      ControlFlowError
    };
    BugType bugType;
    
    enum Severity {
      Critical,
      High,
      Medium,
      Low
    };
    Severity severity;
    
    std::string minimalReproducer;
  };
  
  void reportBug(const interpreter::Program& original,
                const interpreter::Program& mutant,
                const ExecutionTrace& originalTrace,
                const ExecutionTrace& mutantTrace,
                const SemanticValidator::ValidationResult& validation);
  
  void reportCrash(const interpreter::Program& original,
                  const interpreter::Program& mutant,
                  const std::exception& e);
  
private:
  BugReport::TraceDivergence findTraceDivergence(const ExecutionTrace& golden,
                                                 const ExecutionTrace& mutant);
  
  BugReport::BugType classifyBug(const BugReport::TraceDivergence& divergence);
  
  BugReport::Severity assessSeverity(BugReport::BugType type,
                                    const SemanticValidator::ValidationResult& validation);
  
  std::string generateBugId();
  
  void saveBugReport(const BugReport& report);
  
  void logBug(const BugReport& report);
};

// ===== Main Fuzzing Orchestrator =====

struct FuzzingConfig {
  uint32_t threadgroupSize = 32;
  uint32_t waveSize = 32;
  uint32_t maxMutants = 1000;
  uint32_t maxDepth = 5;
  bool enableLogging = true;
  std::string outputDir = "./fuzzing_results";
  std::string seedId;  // Unique identifier for this seed/run
};

class TraceGuidedFuzzer {
private:
  std::vector<std::unique_ptr<MutationStrategy>> mutationStrategies;
  std::unique_ptr<SemanticValidator> validator;
  std::unique_ptr<BugReporter> bugReporter;
  
  // Coverage tracking
  std::set<uint64_t> seenBlockPatterns;
  std::set<uint64_t> seenWavePatterns;
  std::set<uint64_t> seenSyncPatterns;
  
  // Bug tracking
  std::vector<BugReporter::BugReport> bugs;
  
public:
  TraceGuidedFuzzer();
  
  // Version that accepts generation history to only mutate new statements
  interpreter::Program fuzzProgram(const interpreter::Program& seedProgram, 
                  const FuzzingConfig& config,
                  const std::vector<GenerationRound>& history,
                  size_t currentIncrement);
  
private:
  interpreter::Program prepareProgramForMutation(
    const interpreter::Program& program);
    
  std::vector<interpreter::Program> generateMutants(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    const std::vector<GenerationRound>& history,
    size_t currentRound);
  
  std::unique_ptr<interpreter::Statement> applyMutationToStatement(
    const interpreter::Statement* stmt,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    bool& mutationApplied);
  
  // Helper methods for generateMutants refactoring
  std::set<size_t> determineStatementsToMutate(
    const std::vector<GenerationRound>& history,
    size_t currentIncrement);
  
  std::vector<interpreter::Program> applyStatementLevelMutation(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate);
  
  bool hasNewCoverage(const ExecutionTrace& trace);
  
  uint64_t hashBlockPattern(const ExecutionTrace::BlockExecutionRecord& block);
  uint64_t hashWavePattern(const ExecutionTrace::WaveOpRecord& waveOp);
  uint64_t hashSyncPattern(const ExecutionTrace::BarrierRecord& barrier);
  
  void logTrace(const std::string& message, const ExecutionTrace& trace);
  void logMutation(const std::string& message, const std::string& strategy);
  void logSummary(size_t testedMutants, size_t bugsFound);
  
  // Enum to track which mutations were applied
  enum class AppliedMutations : uint32_t {
    None = 0,
    WaveParticipantTracking = 1 << 0,
    LanePermutation = 1 << 1,
    ContextAwareParticipant = 1 << 2,
    WaveParticipantFrequency = 1 << 3,
    WaveParticipantBitTracking = 1 << 4,
    // Add more mutations here as needed
  };
  
  // Enable bitwise operations for AppliedMutations
  friend AppliedMutations operator|(AppliedMutations a, AppliedMutations b) {
    return static_cast<AppliedMutations>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
  }
  friend AppliedMutations& operator|=(AppliedMutations& a, AppliedMutations b) {
    a = a | b;
    return a;
  }
  friend AppliedMutations operator&(AppliedMutations a, AppliedMutations b) {
    return static_cast<AppliedMutations>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
  }
  
  // Helper to apply all mutations in sequence and test the result
  struct MutationResult {
    interpreter::Program mutatedProgram;
    AppliedMutations appliedMutations;
    bool hasMutations() const { return appliedMutations != AppliedMutations::None; }
    std::string getMutationChainString() const;
  };
  
  MutationResult applyAllMutations(
    const interpreter::Program& baseProgram,
    const ExecutionTrace& goldenTrace,
    const std::vector<GenerationRound>* history = nullptr,
    size_t currentIncrement = 0);
};

// ===== LibFuzzer Integration =====







} // namespace fuzzer
} // namespace minihlsl

