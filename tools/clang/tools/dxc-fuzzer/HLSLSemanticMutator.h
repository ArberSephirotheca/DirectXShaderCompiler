#ifndef HLSL_SEMANTIC_MUTATOR_H
#define HLSL_SEMANTIC_MUTATOR_H

#include "HLSLMutationTracker.h"
#include "HLSLProgramGenerator.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <functional>

namespace minihlsl {
namespace fuzzer {

// Semantic-preserving mutations for HLSL programs
class SemanticPreservingMutator {
public:
    // Lane permutation patterns
    enum class PermutationPattern {
        Rotate,      // Rotate lanes by N
        Reverse,     // Reverse lane order
        EvenOddSwap, // Swap even/odd lanes
        Butterfly,   // Butterfly shuffle pattern
        Broadcast,   // Broadcast single lane
        Custom       // Custom permutation
    };
    
private:
    MutationTracker& tracker;
    
    // Generate permutation expression
    std::unique_ptr<interpreter::Expression> generatePermutationExpr(
        PermutationPattern pattern,
        uint32_t participantCount,
        FuzzedDataProvider& provider);
    
    // Helper to wrap expression with WaveReadLaneAt
    std::unique_ptr<interpreter::Expression> wrapWithLanePermutation(
        std::unique_ptr<interpreter::Expression> expr,
        std::unique_ptr<interpreter::Expression> laneExpr);
    
    // Find and replace wave operation in cloned statement
    bool replaceWaveOperation(
        interpreter::Statement* stmt,
        size_t waveOpIndex,
        std::unique_ptr<interpreter::Expression> replacement);
    
    // Generate participant verification code
    std::vector<std::unique_ptr<interpreter::Statement>> generateParticipantVerification(
        const std::string& phase,
        size_t waveOpIndex,
        ProgramState& state);
    
public:
    explicit SemanticPreservingMutator(MutationTracker& t) : tracker(t) {}
    
    // Apply lane permutation mutation
    std::unique_ptr<interpreter::Statement> applyLanePermutation(
        const interpreter::Statement* stmt,
        size_t waveOpIndex,
        ProgramState& state,
        FuzzedDataProvider& provider);
    
    // Apply participant tracking mutation
    std::unique_ptr<interpreter::Statement> applyParticipantTracking(
        const interpreter::Statement* stmt,
        size_t waveOpIndex,
        ProgramState& state,
        FuzzedDataProvider& provider);
    
    // Apply value duplication mutation
    std::unique_ptr<interpreter::Statement> applyValueDuplication(
        const interpreter::Statement* stmt,
        size_t waveOpIndex,
        ProgramState& state,
        FuzzedDataProvider& provider);
    
    // Apply broadcast pattern mutation
    std::unique_ptr<interpreter::Statement> applyBroadcastPattern(
        const interpreter::Statement* stmt,
        size_t waveOpIndex,
        ProgramState& state,
        FuzzedDataProvider& provider);
    
    // Choose appropriate mutation for a wave operation
    MutationType selectMutation(
        const interpreter::Statement* stmt,
        size_t waveOpIndex,
        FuzzedDataProvider& provider);
    
    // Apply best mutation to statement
    std::unique_ptr<interpreter::Statement> mutateStatement(
        const interpreter::Statement* stmt,
        ProgramState& state,
        FuzzedDataProvider& provider);
        
    // Recursive mutation application (similar to MiniHLSLInterpreterFuzzer)
    std::unique_ptr<interpreter::Statement> applyMutationRecursively(
        const interpreter::Statement* stmt,
        ProgramState& state,
        FuzzedDataProvider& provider,
        bool& mutationApplied);
};

// Helper class to find and replace wave operations in AST
class WaveOpReplacer {
private:
    size_t targetWaveOpIndex;
    size_t currentWaveOpCount = 0;
    std::function<std::unique_ptr<interpreter::Expression>(
        std::unique_ptr<interpreter::Expression>)> replacementFunc;
    bool replaced = false;
    
public:
    WaveOpReplacer(size_t index, 
                   std::function<std::unique_ptr<interpreter::Expression>(
                       std::unique_ptr<interpreter::Expression>)> func)
        : targetWaveOpIndex(index), replacementFunc(func) {}
    
    bool wasReplaced() const { return replaced; }
    
    // Replace in expression tree
    std::unique_ptr<interpreter::Expression> 
    replaceInExpression(std::unique_ptr<interpreter::Expression> expr);
    
    // Replace in statement
    std::unique_ptr<interpreter::Statement>
    replaceInStatement(std::unique_ptr<interpreter::Statement> stmt);
};

} // namespace fuzzer
} // namespace minihlsl

#endif // HLSL_SEMANTIC_MUTATOR_H