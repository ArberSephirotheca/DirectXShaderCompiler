#ifndef HLSL_PROGRAM_GENERATOR_H
#define HLSL_PROGRAM_GENERATOR_H

#include "MiniHLSLInterpreter.h"
#include "HLSLMutationTracker.h"
// #include "HLSLSemanticMutator.h" // Not needed - mutations in MiniHLSLInterpreterFuzzer.cpp
#include "MiniHLSLInterpreterFuzzer.h"  // For MutationStrategy and TraceGuidedFuzzer
#include <fuzzer/FuzzedDataProvider.h>
#include <memory>
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
    
    // Pending statement to be added (e.g., loop counter init)
    std::unique_ptr<interpreter::Statement> pendingStatement;
    
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
    void applyMutationsToNew(ProgramState& state, FuzzedDataProvider& provider);
    
    // Apply mutations with sophisticated tracking
    std::unique_ptr<interpreter::Statement> applyMutationsSelectively(
        const interpreter::Statement* stmt,
        ProgramState& state,
        FuzzedDataProvider& provider);
    
    // Handle buffer requirements for mutations
    void handleMutationBufferRequirements(
        const interpreter::Statement* stmt,
        ProgramState& state);
    
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
        enum Type { IF, IF_ELSE, NESTED_IF, FOR_LOOP, WHILE_LOOP, CASCADING_IF_ELSE };
        Type type;
        std::unique_ptr<ParticipantPattern> pattern;
        bool includeBreak;
        bool includeContinue;
        uint32_t nestingDepth;
        // For cascading if-else patterns
        uint32_t numBranches = 3; // Number of if-else-if branches
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
    
    std::unique_ptr<interpreter::Statement>
    generateWhileLoop(const BlockSpec& spec, ProgramState& state,
                      FuzzedDataProvider& provider);
    
    std::unique_ptr<interpreter::Expression>
    generateWaveOperation(ProgramState& state, FuzzedDataProvider& provider);
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