#ifndef HLSL_PROGRAM_GENERATOR_H
#define HLSL_PROGRAM_GENERATOR_H

#include "MiniHLSLInterpreter.h"
#include "HLSLMutationTracker.h"
// #include "HLSLSemanticMutator.h" // Not needed - mutations in MiniHLSLInterpreterFuzzer.cpp
#include "MiniHLSLInterpreterFuzzer.h"  // For MutationStrategy and TraceGuidedFuzzer
#include <fuzzer/FuzzedDataProvider.h>
#include <memory>
#include <optional>
#include <vector>
#include <set>
#include <string>

namespace minihlsl {
namespace fuzzer {

// Information about a generation round
struct GenerationRound {
    size_t roundNumber;
    std::vector<size_t> addedStatementIndices;
    std::vector<MutationType> appliedMutations;
    std::string description;
};

// Core state for incremental program generation
class ProgramState {
public:
    interpreter::Program program;
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

// Forward declarations
class ParticipantPattern;
class ControlFlowGenerator;
class MutationTracker;

// Main incremental generator
class IncrementalGenerator {
private:
    std::unique_ptr<ControlFlowGenerator> cfGenerator;
    std::unique_ptr<MutationTracker> mutationTracker;
    
    std::unique_ptr<ParticipantPattern> createPattern(FuzzedDataProvider& provider);
    void initializeBaseProgram(ProgramState& state, FuzzedDataProvider& provider);
    
public:
    IncrementalGenerator();
    ~IncrementalGenerator();
    
    ProgramState generateIncremental(const uint8_t* data, size_t size);
    
    // Add statements to an existing program
    void addStatementsToProgram(ProgramState& state, const uint8_t* data, size_t size, size_t offset = 0);
};

// Control flow generation
class ControlFlowGenerator {
public:
    struct BlockSpec {
        enum Type { IF, IF_ELSE, NESTED_IF, FOR_LOOP, WHILE_LOOP, CASCADING_IF_ELSE, SWITCH_STMT };
        Type type;
        std::unique_ptr<ParticipantPattern> pattern;
        bool includeBreak;
        bool includeContinue;
        uint32_t nestingDepth;
        // For cascading if-else patterns
        uint32_t numBranches = 3; // Number of if-else-if branches
        
        // Switch-specific configuration
        struct SwitchConfig {
            enum SelectorType {
                LANE_MODULO,      // laneId % N
                VARIABLE_BASED,   // Use existing variable
                THREAD_ID_BASED,  // tid.x % N
            };
            
            SelectorType selectorType = LANE_MODULO;
            uint32_t numCases = 3;        // 2-4 cases
            bool includeDefault = false;  // Add default case
            bool allCasesBreak = true;    // All cases end with break (no fall-through)
        };
        std::optional<SwitchConfig> switchConfig;
        
        // Nesting control
        struct NestingContext {
            uint32_t currentDepth = 0;
            uint32_t maxDepth = 3;
            std::vector<Type> parentTypes;
            std::set<std::string> usedLoopVariables;
            bool allowNesting = false;
        } nestingContext;
    };
    
    std::vector<std::unique_ptr<interpreter::Statement>>
    generateBlock(const BlockSpec& spec, ProgramState& state, 
                  FuzzedDataProvider& provider);
    
private:
    std::unique_ptr<interpreter::Statement> 
    generateIf(const BlockSpec& spec, ProgramState& state, 
               FuzzedDataProvider& provider);
    
    std::unique_ptr<interpreter::Statement>
    generateCascadingIfElse(const BlockSpec& spec, ProgramState& state,
                            FuzzedDataProvider& provider);
    
    std::unique_ptr<interpreter::Statement>
    generateForLoop(const BlockSpec& spec, ProgramState& state,
                    FuzzedDataProvider& provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>>
    generateWhileLoop(const BlockSpec& spec, ProgramState& state,
                      FuzzedDataProvider& provider);
    
    std::unique_ptr<interpreter::Statement>
    generateSwitch(const BlockSpec& spec, ProgramState& state,
                   FuzzedDataProvider& provider);
    
    std::unique_ptr<interpreter::Expression>
    generateWaveOperation(ProgramState& state, FuzzedDataProvider& provider);
    
    // Nesting support
    BlockSpec createNestedBlockSpec(const BlockSpec& parentSpec, 
                                   ProgramState& state,
                                   FuzzedDataProvider& provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>>
    generateNestedBody(const BlockSpec& spec, ProgramState& state,
                      FuzzedDataProvider& provider);
    
    std::string generateLoopVariable(const BlockSpec::NestingContext& context,
                                    BlockSpec::Type loopType,
                                    ProgramState& state);
};

// Utility functions
interpreter::Program prepareProgramForExecution(interpreter::Program program);
std::string serializeProgramToString(const interpreter::Program& program);
void saveBugReport(const uint8_t* data, size_t size, 
                   const interpreter::Program& program,
                   const interpreter::MiniHLSLInterpreter::VerificationResult& verification);

} // namespace fuzzer
} // namespace minihlsl

#endif // HLSL_PROGRAM_GENERATOR_H