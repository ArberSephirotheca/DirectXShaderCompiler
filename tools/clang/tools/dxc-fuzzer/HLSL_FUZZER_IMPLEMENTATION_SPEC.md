# HLSL Fuzzer Implementation Specification

## File Structure

```
MiniHLSLInterpreterFuzzer.h           # Existing - add new generator classes
MiniHLSLInterpreterFuzzer.cpp         # Existing - add implementation
HLSLProgramGenerator.h                # New - generator interfaces
HLSLProgramGenerator.cpp              # New - generator implementation
HLSLParticipantPatterns.h             # New - participant set patterns
HLSLParticipantPatterns.cpp           # New - pattern implementation
fuzz_hlsl_incremental.cpp             # New - libFuzzer entry point
```

## Key Data Structures

### Program State
```cpp
namespace minihlsl {
namespace fuzzer {

struct GenerationRound {
    size_t roundNumber;
    std::vector<size_t> addedStatementIndices;
    std::vector<MutationType> appliedMutations;
    std::string description;
};

struct StatementMetadata {
    size_t originalIndex;
    size_t currentIndex;
    bool isNewlyAdded;
    bool hasMutation;
    MutationType mutationType;
    std::vector<const WaveActiveOp*> waveOps;
};

class ProgramState {
public:
    interpreter::Program program;
    std::vector<StatementMetadata> metadata;
    std::vector<GenerationRound> history;
    
    // Track variable names for generation
    std::set<std::string> declaredVariables;
    uint32_t nextVarIndex = 0;
    
    std::string getNewVariable() {
        std::string varName = "var" + std::to_string(nextVarIndex++);
        declaredVariables.insert(varName);
        return varName;
    }
};

} // namespace fuzzer
} // namespace minihlsl
```

### Participant Pattern Specifications
```cpp
namespace minihlsl {
namespace fuzzer {

class ParticipantPattern {
public:
    virtual ~ParticipantPattern() = default;
    virtual std::unique_ptr<interpreter::Expression> 
        generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) = 0;
    virtual std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) = 0;
    virtual std::string getDescription() const = 0;
};

class SingleLanePattern : public ParticipantPattern {
private:
    uint32_t targetLane;
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override {
        targetLane = provider.ConsumeIntegralInRange<uint32_t>(0, waveSize - 1);
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::WaveGetLaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(targetLane),
            interpreter::BinaryOpExpr::Eq
        );
    }
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override {
        return {targetLane};
    }
    
    std::string getDescription() const override {
        return "Single lane " + std::to_string(targetLane);
    }
};

// Similar implementations for other patterns...

} // namespace fuzzer
} // namespace minihlsl
```

## Generation Algorithm Details

### Control Flow Block Generation
```cpp
class ControlFlowGenerator {
public:
    struct BlockSpec {
        enum Type { IF, IF_ELSE, NESTED_IF, FOR_LOOP, WHILE_LOOP };
        Type type;
        std::unique_ptr<ParticipantPattern> pattern;
        bool includeBreak;
        bool includeContinue;
        uint32_t nestingDepth;
    };
    
    std::vector<std::unique_ptr<interpreter::Statement>>
    generateBlock(const BlockSpec& spec, ProgramState& state, 
                  FuzzedDataProvider& provider) {
        std::vector<std::unique_ptr<interpreter::Statement>> statements;
        
        switch (spec.type) {
            case BlockSpec::IF:
                statements.push_back(generateIf(spec, state, provider));
                break;
            case BlockSpec::FOR_LOOP:
                statements.push_back(generateForLoop(spec, state, provider));
                break;
            // ... other types
        }
        
        return statements;
    }
    
private:
    std::unique_ptr<interpreter::IfStmt> 
    generateIf(const BlockSpec& spec, ProgramState& state, 
               FuzzedDataProvider& provider) {
        // Generate condition using pattern
        auto condition = spec.pattern->generateCondition(
            state.program.preferredWaveSize > 0 ? 
            state.program.preferredWaveSize : 32, provider);
        
        // Generate body with wave operation
        std::vector<std::unique_ptr<interpreter::Statement>> body;
        
        // Always include at least one wave operation
        auto waveOp = generateWaveOperation(state, provider);
        std::string resultVar = state.getNewVariable();
        body.push_back(std::make_unique<interpreter::AssignStmt>(
            resultVar, std::move(waveOp)));
        
        // Optionally add break/continue
        if (spec.includeBreak && provider.ConsumeBool()) {
            body.push_back(std::make_unique<interpreter::BreakStmt>());
        }
        
        return std::make_unique<interpreter::IfStmt>(
            std::move(condition), std::move(body));
    }
    
    std::unique_ptr<interpreter::Expression>
    generateWaveOperation(ProgramState& state, FuzzedDataProvider& provider) {
        // Choose operation type
        auto opType = provider.ConsumeEnum<interpreter::WaveActiveOp::OpType>();
        
        // Choose input - either variable or expression
        std::unique_ptr<interpreter::Expression> input;
        if (!state.declaredVariables.empty() && provider.ConsumeBool()) {
            // Use existing variable
            auto varName = provider.PickValueInArray(
                std::vector<std::string>(state.declaredVariables.begin(), 
                                       state.declaredVariables.end()));
            input = std::make_unique<interpreter::VariableExpr>(varName);
        } else {
            // Use tid.x or literal
            if (provider.ConsumeBool()) {
                input = std::make_unique<interpreter::VariableExpr>("tid.x");
            } else {
                input = std::make_unique<interpreter::LiteralExpr>(
                    provider.ConsumeIntegralInRange<int>(1, 10));
            }
        }
        
        return std::make_unique<interpreter::WaveActiveOp>(
            std::move(input), opType);
    }
};
```

### Incremental Generation Manager
```cpp
class IncrementalGenerator {
private:
    ControlFlowGenerator cfGenerator;
    std::unique_ptr<ParticipantPattern> createPattern(
        FuzzedDataProvider& provider) {
        // Create pattern based on fuzzer input
        enum PatternChoice {
            SINGLE, SPARSE, CONTIGUOUS, DISJOINT, PARITY, ENSURE_NON_EMPTY
        };
        
        auto choice = provider.ConsumeEnum<PatternChoice>();
        switch (choice) {
            case SINGLE:
                return std::make_unique<SingleLanePattern>();
            case SPARSE:
                return std::make_unique<SparseNonContiguousPattern>();
            // ... etc
        }
    }
    
public:
    ProgramState generateIncremental(const uint8_t* data, size_t size) {
        FuzzedDataProvider provider(data, size);
        ProgramState state;
        
        // Initialize base program
        initializeBaseProgram(state, provider);
        
        // Determine number of generation rounds
        uint32_t rounds = provider.ConsumeIntegralInRange<uint32_t>(1, 5);
        
        for (uint32_t round = 0; round < rounds; ++round) {
            // Mark existing statements as not new
            for (auto& meta : state.metadata) {
                meta.isNewlyAdded = false;
            }
            
            // Generate new control flow block
            auto pattern = createPattern(provider);
            ControlFlowGenerator::BlockSpec spec{
                .type = provider.ConsumeEnum<ControlFlowGenerator::BlockSpec::Type>(),
                .pattern = std::move(pattern),
                .includeBreak = provider.ConsumeBool(),
                .includeContinue = provider.ConsumeBool(),
                .nestingDepth = provider.ConsumeIntegralInRange<uint32_t>(0, 2)
            };
            
            auto newStatements = cfGenerator.generateBlock(spec, state, provider);
            
            // Add to program and update metadata
            size_t insertPos = state.program.statements.size();
            for (auto& stmt : newStatements) {
                StatementMetadata meta{
                    .originalIndex = insertPos,
                    .currentIndex = insertPos,
                    .isNewlyAdded = true,
                    .hasMutation = false,
                    .mutationType = MutationType::None,
                    .waveOps = findWaveOps(stmt.get())
                };
                state.metadata.push_back(meta);
                state.program.statements.push_back(std::move(stmt));
                insertPos++;
            }
            
            // Apply mutations to new wave operations only
            applyMutationsToNew(state, provider);
            
            // Record round
            state.history.push_back(GenerationRound{
                .roundNumber = round,
                .addedStatementIndices = /* indices of new statements */,
                .appliedMutations = /* mutations applied */,
                .description = "Round " + std::to_string(round)
            });
        }
        
        return state;
    }
    
private:
    void initializeBaseProgram(ProgramState& state, FuzzedDataProvider& provider) {
        // Set thread configuration
        state.program.numThreadsX = 4; // Or from provider
        state.program.numThreadsY = 1;
        state.program.numThreadsZ = 1;
        
        // Set wave size if specified
        if (provider.ConsumeBool()) {
            state.program.preferredWaveSize = provider.ConsumeBool() ? 32 : 64;
        }
        
        // Add SV_DispatchThreadID parameter
        interpreter::ParameterSig tidParam{
            .name = "tid",
            .type = interpreter::HLSLType::Uint3,
            .semantic = interpreter::HLSLSemantic::SV_DispatchThreadID
        };
        state.program.entryInputs.parameters.push_back(tidParam);
        state.program.entryInputs.hasDispatchThreadID = true;
        
        // Add basic variable declarations
        state.program.statements.push_back(
            std::make_unique<interpreter::VarDeclStmt>(
                "result", interpreter::HLSLType::Uint,
                std::make_unique<interpreter::LiteralExpr>(0)));
        state.declaredVariables.insert("result");
    }
    
    void applyMutationsToNew(ProgramState& state, FuzzedDataProvider& provider) {
        for (size_t i = 0; i < state.metadata.size(); ++i) {
            auto& meta = state.metadata[i];
            if (meta.isNewlyAdded && !meta.hasMutation && !meta.waveOps.empty()) {
                // Decide which mutation to apply
                bool useLanePermutation = provider.ConsumeBool();
                
                if (useLanePermutation) {
                    // Apply lane permutation
                    LanePermutationMutation mutation;
                    // ... apply mutation
                    meta.hasMutation = true;
                    meta.mutationType = MutationType::LanePermutation;
                } else {
                    // Apply participant tracking
                    WaveParticipantTrackingMutation mutation;
                    // ... apply mutation
                    meta.hasMutation = true;
                    meta.mutationType = MutationType::ParticipantTracking;
                }
            }
        }
    }
};
```

## libFuzzer Entry Point

```cpp
// fuzz_hlsl_incremental.cpp
#include "HLSLProgramGenerator.h"
#include "MiniHLSLInterpreter.h"
#include <fuzzer/FuzzedDataProvider.h>

using namespace minihlsl;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Minimum size for meaningful generation
    if (Size < 16) return 0;
    
    try {
        // Generate program
        fuzzer::IncrementalGenerator generator;
        auto state = generator.generateIncremental(Data, Size);
        
        // Prepare for execution
        auto program = fuzzer::prepareProgramForExecution(state.program);
        
        // Execute with interpreter
        interpreter::MiniHLSLInterpreter interpreter;
        uint32_t waveSize = program.preferredWaveSize > 0 ? 
                           program.preferredWaveSize : 32;
        
        auto verification = interpreter.verifyProgram(program, waveSize);
        
        // Check for bugs
        if (!verification.isOrderIndependent) {
            // Log the bug
            std::cerr << "Found order-dependent behavior!" << std::endl;
            std::cerr << "Program:\n" << 
                fuzzer::serializeProgramToString(program) << std::endl;
            
            // Save to file
            fuzzer::saveBugReport(Data, Size, program, verification);
            
            // Optionally crash for fuzzer
            if (getenv("FUZZ_CRASH_ON_BUG")) {
                abort();
            }
        }
        
    } catch (const std::exception& e) {
        // Ignore generation/execution errors
        return 0;
    }
    
    return 0;
}

// Optional: Custom mutator for better fuzzing
extern "C" size_t LLVMFuzzerCustomMutator(
    uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
    // Can implement smart mutations that preserve program structure
    return Size;
}
```

## Build Configuration

```cmake
# Add to CMakeLists.txt
add_executable(fuzz-hlsl-incremental
    fuzz_hlsl_incremental.cpp
    HLSLProgramGenerator.cpp
    HLSLParticipantPatterns.cpp
    MiniHLSLInterpreterFuzzer.cpp
    MiniHLSLInterpreter.cpp
    MiniHLSLInterpreterTraceCapture.cpp
)

target_compile_options(fuzz-hlsl-incremental PRIVATE
    -g -O1 -fsanitize=fuzzer,address
)

target_link_libraries(fuzz-hlsl-incremental PRIVATE
    -fsanitize=fuzzer,address
    LLVMSupport
    clangAST
)
```

## Testing Strategy

### Unit Tests
1. Test each participant pattern generates valid conditions
2. Test incremental generation produces valid programs
3. Test mutations are only applied to new code
4. Test wave operations are always under divergent control

### Integration Tests
1. Generate 1000 random programs, verify all are valid
2. Check participant patterns produce expected lane sets
3. Verify incremental generation history is accurate
4. Test libFuzzer integration with small corpus

### Fuzzing Campaigns
1. Initial run: 1 hour with basic corpus
2. Extended run: 24 hours with coverage guidance
3. Targeted runs: Focus on specific patterns (loops, nested ifs, etc.)
4. Wave size comparison: Run same corpus with waveSize=32 and 64

## Performance Considerations

1. **Caching**: Cache generated base programs for common prefixes
2. **Fast Generation**: Pre-compile pattern generators
3. **Efficient Execution**: Skip redundant verifications
4. **Memory Management**: Reuse AST nodes where possible