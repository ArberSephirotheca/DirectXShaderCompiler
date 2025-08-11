#ifndef HLSL_PROGRAM_GENERATOR_H
#define HLSL_PROGRAM_GENERATOR_H

#include "MiniHLSLInterpreter.h"
#include "HLSLMutationTracker.h"
#include "HLSLSemanticMutator.h"
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
class SemanticPreservingMutator;

// Main incremental generator
class IncrementalGenerator {
private:
    std::unique_ptr<ControlFlowGenerator> cfGenerator;
    std::unique_ptr<MutationTracker> mutationTracker;
    std::unique_ptr<SemanticPreservingMutator> semanticMutator;
    
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
};

// Control flow generation
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
                  FuzzedDataProvider& provider);
    
private:
    std::unique_ptr<interpreter::Statement> 
    generateIf(const BlockSpec& spec, ProgramState& state, 
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