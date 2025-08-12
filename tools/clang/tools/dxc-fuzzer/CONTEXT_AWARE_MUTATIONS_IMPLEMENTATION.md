# Context-Aware Mutations Implementation Guide

## Overview

This document provides detailed implementation guidance for adding context-aware mutations to the MiniHLSL fuzzer. These mutations leverage dynamic execution blocks to target specific loop iterations and control flow contexts.

## 1. Context-Aware Participant Mutation

### Purpose
Add lanes to specific loop iterations by generating conditions like `(i == 1 && laneId == 3)` that target exact dynamic block instances.

### Header Definition
```cpp
// Add to MiniHLSLInterpreterFuzzer.h

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
```

### Implementation
```cpp
// In MiniHLSLInterpreterFuzzer.cpp

bool ContextAwareParticipantMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Can apply to wave operations inside loops
  bool hasWaveOp = false;
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    hasWaveOp = dynamic_cast<const interpreter::WaveActiveOp*>(
      assign->getExpression()) != nullptr;
  } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    hasWaveOp = varDecl->getInit() && 
      dynamic_cast<const interpreter::WaveActiveOp*>(varDecl->getInit()) != nullptr;
  }
  
  if (!hasWaveOp) return false;
  
  // Check if this wave op executed inside a loop
  for (const auto& waveOp : trace.waveOperations) {
    if (waveOp.blockId > 100) { // Simple heuristic: high block IDs indicate loop iterations
      return true;
    }
  }
  
  return false;
}

std::vector<ContextAwareParticipantMutation::IterationContext> 
ContextAwareParticipantMutation::extractIterationContexts(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  std::vector<IterationContext> contexts;
  
  // Find all wave operations that correspond to this statement
  for (const auto& waveOp : trace.waveOperations) {
    // Match by operation type and block context
    bool matches = false;
    
    // Check if this wave op came from our statement
    if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
      if (auto* waveExpr = dynamic_cast<const interpreter::WaveActiveOp*>(
          assign->getExpression())) {
        matches = (waveOp.waveOpEnumType == static_cast<int>(waveExpr->getOpType()));
      }
    }
    
    if (!matches) continue;
    
    // Extract iteration context from block ID
    // Block ID encoding: base + iteration * 100
    // E.g., block 234 = base 34 + iteration 2 * 100
    uint32_t blockId = waveOp.blockId;
    if (blockId < 100) continue; // Not in a loop
    
    IterationContext ctx;
    ctx.blockId = blockId;
    ctx.waveId = waveOp.waveId;
    ctx.existingParticipants = waveOp.arrivedParticipants;
    
    // Decode iteration value
    uint32_t baseBlockId = blockId % 100;
    ctx.iterationValue = (blockId - baseBlockId) / 100;
    
    // Find loop variable by traversing parent blocks
    ctx.loopVariable = findLoopVariable(blockId, trace);
    
    if (!ctx.loopVariable.empty()) {
      contexts.push_back(ctx);
    }
  }
  
  return contexts;
}

std::string ContextAwareParticipantMutation::findLoopVariable(
    uint32_t blockId,
    const ExecutionTrace& trace) const {
  
  // Traverse parent blocks to find the loop statement
  uint32_t currentId = blockId;
  
  while (currentId > 0 && trace.blocks.count(currentId)) {
    const auto& block = trace.blocks.at(currentId);
    
    // Check if source statement is a loop
    if (block.sourceStatement) {
      if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(
          block.sourceStatement)) {
        return forStmt->getLoopVariable();
      }
      // For while loops, we'd need to track the counter variable differently
    }
    
    currentId = block.parentBlockId;
  }
  
  // Fallback: assume common names
  if (blockId >= 100 && blockId < 200) return "i0";
  if (blockId >= 200 && blockId < 300) return "i0"; // Same var, different iteration
  
  return "";
}

std::unique_ptr<interpreter::Statement> 
ContextAwareParticipantMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Extract all iteration contexts where this wave op executed
  auto contexts = extractIterationContexts(stmt, trace);
  if (contexts.empty()) {
    return stmt->clone();
  }
  
  // Choose a context to mutate
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> ctxDist(0, contexts.size() - 1);
  const auto& targetContext = contexts[ctxDist(rng)];
  
  // Find lanes that didn't participate in this iteration
  std::set<interpreter::LaneId> availableLanes;
  uint32_t waveSize = trace.threadHierarchy.waveSize;
  
  for (uint32_t lane = 0; lane < waveSize; ++lane) {
    if (targetContext.existingParticipants.find(lane) == 
        targetContext.existingParticipants.end()) {
      availableLanes.insert(lane);
    }
  }
  
  if (availableLanes.empty()) {
    return stmt->clone(); // All lanes already participate
  }
  
  // Choose a new lane to add
  std::vector<interpreter::LaneId> laneVec(availableLanes.begin(), 
                                           availableLanes.end());
  std::uniform_int_distribution<size_t> laneDist(0, laneVec.size() - 1);
  interpreter::LaneId newLane = laneVec[laneDist(rng)];
  
  // Create the iteration-specific condition
  auto condition = createIterationSpecificCondition(targetContext, newLane);
  
  // Create the mutated statement structure
  // We'll create: if (original_condition) { wave_op } 
  //               else if (iter == X && lane == Y) { wave_op }
  
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // First, the original statement
  statements.push_back(stmt->clone());
  
  // Then, add our context-specific branch
  std::vector<std::unique_ptr<interpreter::Statement>> contextBranch;
  contextBranch.push_back(stmt->clone()); // Same wave operation
  
  auto contextIf = std::make_unique<interpreter::IfStmt>(
    std::move(condition),
    std::move(contextBranch),
    std::vector<std::unique_ptr<interpreter::Statement>>{}
  );
  
  statements.push_back(std::move(contextIf));
  
  // Wrap in a compound statement
  auto compound = std::make_unique<interpreter::CompoundStmt>(std::move(statements));
  
  return compound;
}

std::unique_ptr<interpreter::Expression>
ContextAwareParticipantMutation::createIterationSpecificCondition(
    const IterationContext& context,
    interpreter::LaneId newLane) const {
  
  // Create: (loopVar == iteration) && (laneId == newLane)
  
  // loopVar == iteration
  auto loopVarExpr = std::make_unique<interpreter::VariableExpr>(context.loopVariable);
  auto iterValue = std::make_unique<interpreter::LiteralExpr>(context.iterationValue);
  auto iterCheck = std::make_unique<interpreter::BinaryOpExpr>(
    std::move(loopVarExpr), std::move(iterValue), 
    interpreter::BinaryOpExpr::Eq);
  
  // laneId == newLane  
  auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto laneValue = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(newLane));
  auto laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
    std::move(laneExpr), std::move(laneValue),
    interpreter::BinaryOpExpr::Eq);
  
  // Combine with &&
  return std::make_unique<interpreter::BinaryOpExpr>(
    std::move(iterCheck), std::move(laneCheck),
    interpreter::BinaryOpExpr::And);
}

bool ContextAwareParticipantMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  
  // The mutation adds participants but doesn't change the operation
  // For associative operations, this should preserve semantics
  // TODO: Implement full validation
  return true;
}
```

## 2. Block-Specific Mutation Tracking

### Enhanced Program State
```cpp
// In HLSLProgramGenerator.h

struct AppliedMutation {
  MutationType type;
  std::string description;
  size_t statementIndex;
  uint32_t targetBlockId;
  std::chrono::system_clock::time_point timestamp;
};

class ProgramState {
public:
  // Existing members...
  
  // NEW: Track mutations per dynamic block
  std::map<uint32_t, std::vector<AppliedMutation>> blockMutations;
  
  void recordMutation(uint32_t blockId, const AppliedMutation& mutation) {
    blockMutations[blockId].push_back(mutation);
  }
  
  bool hasBlockBeenMutated(uint32_t blockId, MutationType type) const {
    auto it = blockMutations.find(blockId);
    if (it == blockMutations.end()) return false;
    
    return std::any_of(it->second.begin(), it->second.end(),
      [type](const auto& mut) { return mut.type == type; });
  }
  
  std::vector<uint32_t> getUnmutatedBlocks(const ExecutionTrace& trace) const {
    std::vector<uint32_t> unmutated;
    
    for (const auto& [blockId, blockRecord] : trace.blocks) {
      if (blockMutations.find(blockId) == blockMutations.end()) {
        unmutated.push_back(blockId);
      }
    }
    
    return unmutated;
  }
};
```

### Block-Aware Fuzzer Enhancement
```cpp
// Add to TraceGuidedFuzzer in MiniHLSLInterpreterFuzzer.cpp

class BlockAwareTrace : public ExecutionTrace {
public:
  BlockAwareTrace(const ExecutionTrace& base, uint32_t targetBlock) 
    : ExecutionTrace(base), targetBlockId(targetBlock) {}
  
  uint32_t getTargetBlockId() const { return targetBlockId; }
  
  // Filter wave operations to only those in target block
  std::vector<WaveOpRecord> getTargetBlockWaveOps() const {
    std::vector<WaveOpRecord> filtered;
    for (const auto& op : waveOperations) {
      if (op.blockId == targetBlockId) {
        filtered.push_back(op);
      }
    }
    return filtered;
  }
  
private:
  uint32_t targetBlockId;
};

std::unique_ptr<interpreter::Statement> 
TraceGuidedFuzzer::generateMutationForBlock(
    uint32_t targetBlockId,
    const ExecutionTrace::BlockExecutionRecord& blockRecord,
    const ExecutionTrace& trace,
    MutationStrategy* strategy,
    ProgramState* state) {
  
  // Create block-specific view of trace
  BlockAwareTrace blockTrace(trace, targetBlockId);
  
  // Find the statement that created this block
  const auto* sourceStmt = blockRecord.sourceStatement;
  if (!sourceStmt) return nullptr;
  
  // Check if strategy can apply to this specific block
  if (!strategy->canApply(sourceStmt, blockTrace)) {
    return nullptr;
  }
  
  // Generate mutation
  auto mutation = strategy->apply(sourceStmt, blockTrace);
  
  if (mutation && state) {
    // Record the mutation
    AppliedMutation record;
    record.type = strategy->getType();
    record.targetBlockId = targetBlockId;
    record.description = strategy->getName() + " targeting block " + 
                        std::to_string(targetBlockId);
    record.timestamp = std::chrono::system_clock::now();
    
    state->recordMutation(targetBlockId, record);
  }
  
  return mutation;
}
```

## 3. Loop-Aware Pattern Generation

### Pattern Hierarchy
```cpp
// In HLSLParticipantPatterns.h

// Base class enhancement
class ParticipantPattern {
public:
  // Existing method for non-loop contexts
  virtual std::unique_ptr<interpreter::Expression> generateCondition(
      uint32_t waveSize,
      FuzzedDataProvider& provider) const = 0;
  
  // NEW: Loop-aware generation with iteration context
  virtual std::unique_ptr<interpreter::Expression> generateConditionForIteration(
      uint32_t waveSize,
      int loopIteration,
      const std::string& loopVariable,
      FuzzedDataProvider& provider) const {
    // Default: ignore iteration, use base generation
    return generateCondition(waveSize, provider);
  }
};

// Loop-aware pattern that changes per iteration
class IterationModuloPattern : public ParticipantPattern {
public:
  std::unique_ptr<interpreter::Expression> generateCondition(
      uint32_t waveSize,
      FuzzedDataProvider& provider) const override {
    // For non-loop usage
    uint32_t modulo = provider.ConsumeIntegralInRange<uint32_t>(2, 4);
    uint32_t remainder = provider.ConsumeIntegralInRange<uint32_t>(0, modulo - 1);
    
    return createModuloCondition(modulo, remainder);
  }
  
  std::unique_ptr<interpreter::Expression> generateConditionForIteration(
      uint32_t waveSize,
      int loopIteration,
      const std::string& loopVariable,
      FuzzedDataProvider& provider) const override {
    
    // Different pattern per iteration
    switch (loopIteration % 3) {
      case 0:
        // Iteration 0: lanes where (laneId % 3) == 0
        return createModuloCondition(3, 0);
        
      case 1:
        // Iteration 1: lanes where (laneId % 3) == 1
        return createModuloCondition(3, 1);
        
      case 2:
        // Iteration 2: all even lanes
        return createModuloCondition(2, 0);
    }
    
    return createModuloCondition(2, 0);
  }
  
private:
  std::unique_ptr<interpreter::Expression> createModuloCondition(
      uint32_t modulo, uint32_t remainder) const {
    
    // (laneId % modulo) == remainder
    auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
    auto modValue = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(modulo));
    auto modExpr = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(laneExpr), std::move(modValue), interpreter::BinaryOpExpr::Mod);
    
    auto remValue = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(remainder));
    return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(modExpr), std::move(remValue), interpreter::BinaryOpExpr::Eq);
  }
};

// Pattern that creates iteration-specific single lane participation
class IterationSingleLanePattern : public ParticipantPattern {
public:
  std::unique_ptr<interpreter::Expression> generateCondition(
      uint32_t waveSize,
      FuzzedDataProvider& provider) const override {
    
    uint32_t lane = provider.ConsumeIntegralInRange<uint32_t>(0, waveSize - 1);
    return createLaneCheck(lane);
  }
  
  std::unique_ptr<interpreter::Expression> generateConditionForIteration(
      uint32_t waveSize,
      int loopIteration,
      const std::string& loopVariable,
      FuzzedDataProvider& provider) const override {
    
    // Lane matches iteration number (mod waveSize)
    uint32_t targetLane = loopIteration % waveSize;
    return createLaneCheck(targetLane);
  }
  
private:
  std::unique_ptr<interpreter::Expression> createLaneCheck(uint32_t lane) const {
    auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
    auto laneVal = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(lane));
    return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(laneExpr), std::move(laneVal), interpreter::BinaryOpExpr::Eq);
  }
};
```

### Updated Control Flow Generator
```cpp
// In HLSLProgramGenerator.cpp

std::unique_ptr<interpreter::Statement>
ControlFlowGenerator::generateForLoop(const BlockSpec& spec, ProgramState& state,
                                     FuzzedDataProvider& provider) {
  
  // Initialize loop variable
  std::string loopVar = "i" + std::to_string(state.nextVarIndex++);
  state.declaredVariables.insert(loopVar);
  
  auto initExpr = std::make_unique<interpreter::LiteralExpr>(0);
  
  uint32_t loopCount = provider.ConsumeIntegralInRange<uint32_t>(2, 3);
  auto condition = std::make_unique<interpreter::BinaryOpExpr>(
    std::make_unique<interpreter::VariableExpr>(loopVar),
    std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(loopCount)),
    interpreter::BinaryOpExpr::Lt);
  
  auto increment = std::make_unique<interpreter::AssignExpr>(
    loopVar,
    std::make_unique<interpreter::BinaryOpExpr>(
      std::make_unique<interpreter::VariableExpr>(loopVar),
      std::make_unique<interpreter::LiteralExpr>(1),
      interpreter::BinaryOpExpr::Add));
  
  // Generate iteration-aware body
  std::vector<std::unique_ptr<interpreter::Statement>> body;
  
  // Check if pattern supports iteration-aware generation
  bool useIterationAware = provider.ConsumeBool();
  
  if (useIterationAware && spec.pattern) {
    // Generate different patterns for different iterations
    for (uint32_t iter = 0; iter < loopCount; ++iter) {
      // Create iteration-specific condition
      auto iterCheck = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(loopVar),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int>(iter)),
        interpreter::BinaryOpExpr::Eq);
      
      // Get pattern for this iteration
      auto participantCond = spec.pattern->generateConditionForIteration(
        state.program.waveSize > 0 ? state.program.waveSize : 32,
        iter, loopVar, provider);
      
      // Combine: if (i == iter && participantCondition)
      auto combined = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(iterCheck), std::move(participantCond),
        interpreter::BinaryOpExpr::And);
      
      // Create wave operation for this iteration
      std::vector<std::unique_ptr<interpreter::Statement>> iterBody;
      auto waveOp = generateWaveOperation(state, provider);
      iterBody.push_back(std::make_unique<interpreter::AssignStmt>(
        "result", std::move(waveOp)));
      
      body.push_back(std::make_unique<interpreter::IfStmt>(
        std::move(combined), std::move(iterBody)));
    }
  } else {
    // Use original non-iteration-aware generation
    auto participantCondition = spec.pattern->generateCondition(
      state.program.waveSize > 0 ? state.program.waveSize : 32, provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
    auto waveOp = generateWaveOperation(state, provider);
    waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
      "result", std::move(waveOp)));
    
    body.push_back(std::make_unique<interpreter::IfStmt>(
      std::move(participantCondition), std::move(waveBody)));
  }
  
  return std::make_unique<interpreter::ForStmt>(
    loopVar, std::move(initExpr), std::move(condition),
    std::move(increment), std::move(body));
}
```

## Integration Steps

### 1. Register New Mutation Strategy
```cpp
// In TraceGuidedFuzzer constructor
TraceGuidedFuzzer::TraceGuidedFuzzer() {
  // Existing strategies
  mutationStrategies.push_back(std::make_unique<LanePermutationMutation>());
  mutationStrategies.push_back(std::make_unique<WaveParticipantTrackingMutation>());
  
  // NEW: Context-aware mutation
  mutationStrategies.push_back(std::make_unique<ContextAwareParticipantMutation>());
}
```

### 2. Update Pattern Factory
```cpp
// In HLSLParticipantPatterns.cpp
std::unique_ptr<ParticipantPattern> createLoopAwarePattern(FuzzedDataProvider& provider) {
  
  uint32_t choice = provider.ConsumeIntegralInRange<uint32_t>(0, 3);
  
  switch (choice) {
    case 0:
      return std::make_unique<IterationSingleLanePattern>();
    case 1:
      return std::make_unique<IterationModuloPattern>();
    case 2:
      return std::make_unique<IterationParityPattern>();
    case 3:
      return std::make_unique<IterationRangePattern>();
    default:
      return std::make_unique<IterationSingleLanePattern>();
  }
}
```

### 3. Pass State Through Pipeline
```cpp
// Modify IncrementalFuzzingPipeline::testMutations signature
PipelineResult::IncrementResult IncrementalFuzzingPipeline::testMutations(
    const interpreter::Program& program,
    const ExecutionTrace& goldenTrace,
    interpreter::Program& mutatedProgram,
    ProgramState& state) { // Now non-const reference
  
  // Pass state to fuzzer for mutation tracking
  mutatedProgram = fuzzer->fuzzProgram(program, fuzzConfig, 
                                       state.history, currentRound,
                                       &state);
}
```

## Testing Strategy

### 1. Unit Tests for Context Extraction
```cpp
TEST(ContextAwareMutation, ExtractIterationContext) {
  // Create a trace with known block structure
  ExecutionTrace trace;
  
  // Add block 134 (iteration 1, base block 34)
  ExecutionTrace::BlockExecutionRecord block134;
  block134.blockId = 134;
  block134.blockType = interpreter::BlockType::IfThen;
  trace.blocks[134] = block134;
  
  // Add wave operation in this block
  ExecutionTrace::WaveOpRecord waveOp;
  waveOp.blockId = 134;
  waveOp.waveOpEnumType = static_cast<int>(interpreter::WaveActiveOp::Sum);
  waveOp.arrivedParticipants = {1}; // Only lane 1
  trace.waveOperations.push_back(waveOp);
  
  // Test extraction
  ContextAwareParticipantMutation mutation;
  auto contexts = mutation.extractIterationContexts(stmt, trace);
  
  ASSERT_EQ(contexts.size(), 1);
  EXPECT_EQ(contexts[0].iterationValue, 1);
  EXPECT_EQ(contexts[0].blockId, 134);
}
```

### 2. Integration Test
```cpp
TEST(ContextAwareMutation, AddLaneToSpecificIteration) {
  // Generate program with loop
  const char* programSrc = R"(
    for (int i = 0; i < 3; i++) {
      if (laneId == i) {
        result = WaveActiveSum(value);
      }
    }
  )";
  
  // Execute and capture trace
  auto trace = executeAndTrace(program);
  
  // Apply context-aware mutation
  ContextAwareParticipantMutation mutation;
  auto mutated = mutation.apply(findWaveOpStmt(program), trace);
  
  // Verify mutation adds lane to specific iteration
  // Should contain: else if (i == 1 && laneId == 3)
  EXPECT_TRUE(containsIterationSpecificBranch(mutated, 1, 3));
}
```

## Expected Outcomes

1. **Precise Targeting**: Mutations can add participants to specific loop iterations
2. **Better Coverage**: Explore participant patterns that vary by iteration
3. **Reduced False Positives**: Context-aware mutations maintain semantic equivalence
4. **Improved Bug Finding**: Detect iteration-specific wave operation bugs

## Performance Considerations

1. **Block ID Encoding**: Current scheme limits to 99 base blocks and 99 iterations
2. **Trace Size**: Context tracking increases trace memory usage
3. **Mutation Explosion**: Limit mutations per block to avoid combinatorial explosion

## Future Enhancements

1. **Nested Loop Support**: Handle multiple loop variables
2. **While Loop Support**: Track iteration count for while loops
3. **Break/Continue Awareness**: Handle early exits in mutation generation
4. **Cross-Iteration Dependencies**: Detect and preserve data dependencies