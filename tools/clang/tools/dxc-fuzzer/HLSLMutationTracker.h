#ifndef HLSL_MUTATION_TRACKER_H
#define HLSL_MUTATION_TRACKER_H

#include "MiniHLSLInterpreter.h"
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <string>

namespace minihlsl {
namespace fuzzer {

// Forward declarations
class ProgramState;

// Detailed mutation types
enum class MutationType {
    None,
    LanePermutation,      // Reorders lane values
    ParticipantTracking,  // Adds verification code
    ValueDuplication,     // Duplicates values across lanes
    BroadcastPattern,     // Broadcasts single lane to others
};

// Location of a wave operation in the AST
struct WaveOpLocation {
    const interpreter::WaveActiveOp* op;
    std::vector<size_t> path;  // Path indices to reach this op in AST
    bool isMutated;
    MutationType appliedMutation;
    
    WaveOpLocation(const interpreter::WaveActiveOp* o, std::vector<size_t> p)
        : op(o), path(std::move(p)), isMutated(false), appliedMutation(MutationType::None) {}
};

// Record of a single mutation
struct MutationRecord {
    MutationType type;
    size_t waveOpIndex;      // Which wave op in the statement
    size_t round;            // Generation round when applied
    std::string description;
    
    // Additional mutation-specific data
    struct LanePermutationData {
        enum Pattern { Rotate, Reverse, EvenOdd, Custom } pattern;
        int offset;  // For rotate pattern
    };
    
    std::variant<std::monostate, LanePermutationData> mutationData;
};

// Enhanced metadata for tracking mutations
struct StatementMetadata {
    size_t originalIndex;
    size_t currentIndex;
    bool isNewlyAdded;
    size_t generationRound;
    
    // Mutation tracking
    std::set<MutationType> appliedMutations;
    std::vector<MutationRecord> mutationHistory;
    
    // Wave operation tracking
    std::vector<WaveOpLocation> waveOps;
    std::set<size_t> mutatedWaveOpIndices;
    
    // Variable dependencies
    std::set<std::string> readsVariables;
    std::set<std::string> writesVariables;
    
    // Control flow context
    enum Context { TopLevel, IfBlock, ElseBlock, ForLoop, WhileLoop } context;
    size_t nestingLevel;
    
    StatementMetadata() : originalIndex(0), currentIndex(0), isNewlyAdded(false),
                         generationRound(0), context(TopLevel), nestingLevel(0) {}
};

// Tracks mutations across the entire program
class MutationTracker {
private:
    // Maps statement pointers to their metadata
    std::map<const interpreter::Statement*, StatementMetadata> metadataMap;
    
    // Global tracking of mutated operations
    std::set<const interpreter::WaveActiveOp*> globallyMutatedOps;
    
    // Current generation round
    size_t currentRound;
    
public:
    MutationTracker() : currentRound(0) {}
    
    // Check if a mutation can be applied
    bool canApplyMutation(const interpreter::Statement* stmt,
                         MutationType mutation,
                         size_t waveOpIndex) const;
    
    // Record a mutation
    void recordMutation(const interpreter::Statement* stmt,
                       MutationType mutation,
                       size_t waveOpIndex,
                       const MutationRecord& record);
    
    // Register a new statement
    void registerStatement(const interpreter::Statement* stmt,
                          const StatementMetadata& metadata);
    
    // Update round
    void advanceRound() { currentRound++; }
    size_t getCurrentRound() const { return currentRound; }
    
    // Get metadata for a statement
    const StatementMetadata* getMetadata(const interpreter::Statement* stmt) const;
    StatementMetadata* getMutableMetadata(const interpreter::Statement* stmt);
    
    // Check mutation compatibility
    bool areMutationsCompatible(MutationType existing, MutationType proposed) const;
    
    // Find unmutated wave operations in a statement
    std::vector<size_t> findUnmutatedWaveOps(const interpreter::Statement* stmt) const;
    
    // Check if statement is from current round
    bool isFromCurrentRound(const interpreter::Statement* stmt) const;
};

// Utility functions for AST traversal
std::vector<WaveOpLocation> findAllWaveOps(const interpreter::Statement* stmt);
std::vector<WaveOpLocation> findAllWaveOps(const interpreter::Expression* expr,
                                          std::vector<size_t> currentPath = {});

// Clone statements while preserving structure
std::unique_ptr<interpreter::Statement> cloneStatement(const interpreter::Statement* stmt);
std::unique_ptr<interpreter::Expression> cloneExpression(const interpreter::Expression* expr);

} // namespace fuzzer
} // namespace minihlsl

#endif // HLSL_MUTATION_TRACKER_H