#include "HLSLSemanticMutator.h"
#include <sstream>
#include <cassert>

namespace minihlsl {
namespace fuzzer {

std::unique_ptr<interpreter::Expression> 
SemanticPreservingMutator::generatePermutationExpr(
    PermutationPattern pattern,
    uint32_t participantCount,
    FuzzedDataProvider& provider) {
    
    auto laneIndex = std::make_unique<interpreter::LaneIndexExpr>();
    
    switch (pattern) {
        case PermutationPattern::Rotate: {
            // (lane + offset) % count
            uint32_t offset = provider.ConsumeIntegralInRange<uint32_t>(1, participantCount - 1);
            auto offsetExpr = std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(offset));
            auto countExpr = std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(participantCount));
            
            auto sum = std::make_unique<interpreter::BinaryOpExpr>(
                std::move(laneIndex),
                std::move(offsetExpr),
                interpreter::BinaryOpExpr::OpType::Add
            );
            
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(sum),
                std::move(countExpr),
                interpreter::BinaryOpExpr::OpType::Mod
            );
        }
        
        case PermutationPattern::Reverse: {
            // (count - 1) - lane
            auto countMinus1 = std::make_unique<interpreter::LiteralExpr>(
                static_cast<int32_t>(participantCount - 1)
            );
            
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(countMinus1),
                std::move(laneIndex),
                interpreter::BinaryOpExpr::OpType::Sub
            );
        }
        
        case PermutationPattern::EvenOddSwap: {
            // lane ^ 1 (swaps bit 0)
            auto one = std::make_unique<interpreter::LiteralExpr>(1);
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(laneIndex),
                std::move(one),
                interpreter::BinaryOpExpr::OpType::Xor
            );
        }
        
        case PermutationPattern::Butterfly: {
            // lane ^ (1 << stage)
            uint32_t stage = provider.ConsumeIntegralInRange<uint32_t>(0, 4);
            auto mask = std::make_unique<interpreter::LiteralExpr>(1 << stage);
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(laneIndex),
                std::move(mask),
                interpreter::BinaryOpExpr::OpType::Xor
            );
        }
        
        case PermutationPattern::Broadcast: {
            // Always read from lane 0 (or any fixed lane)
            uint32_t sourceLane = provider.ConsumeIntegralInRange<uint32_t>(0, participantCount - 1);
            return std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(sourceLane));
        }
        
        case PermutationPattern::Custom:
        default: {
            // Random valid lane
            uint32_t targetLane = provider.ConsumeIntegralInRange<uint32_t>(0, participantCount - 1);
            return std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(targetLane));
        }
    }
}

std::unique_ptr<interpreter::Expression>
SemanticPreservingMutator::wrapWithLanePermutation(
    std::unique_ptr<interpreter::Expression> expr,
    std::unique_ptr<interpreter::Expression> laneExpr) {
    
    return std::make_unique<interpreter::WaveReadLaneAt>(
        std::move(expr),
        std::move(laneExpr)
    );
}

bool SemanticPreservingMutator::replaceWaveOperation(
    interpreter::Statement* stmt,
    size_t waveOpIndex,
    std::unique_ptr<interpreter::Expression> replacement) {
    
    // TODO: Implement proper wave operation replacement
    // For now, we'll need to recreate the entire statement
    // This is a simplified implementation
    return false;
}

std::vector<std::unique_ptr<interpreter::Statement>>
SemanticPreservingMutator::generateParticipantVerification(
    const std::string& phase,
    size_t waveOpIndex,
    ProgramState& state) {
    
    std::vector<std::unique_ptr<interpreter::Statement>> verification;
    
    // Generate unique variable names
    std::string maskVar = "_verify_mask_" + phase + "_" + std::to_string(waveOpIndex);
    std::string checkVar = "_verify_check_" + phase + "_" + std::to_string(waveOpIndex);
    
    // uint4 mask = WaveActiveBallot(true);
    verification.push_back(std::make_unique<interpreter::VarDeclStmt>(
        maskVar,
        interpreter::HLSLType::Uint4,
        std::make_unique<interpreter::WaveActiveOp>(
            std::make_unique<interpreter::LiteralExpr>(1),  // true
            interpreter::WaveActiveOp::OpType::Ballot
        )
    ));
    
    // Store to buffer for verification: _participant_check_sum[tid.x + offset] = mask
    verification.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
        "_participant_check_sum",
        std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::DispatchThreadIdExpr>(0),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(waveOpIndex * 2 + (phase == "pre" ? 0 : 1))),
            interpreter::BinaryOpExpr::OpType::Add
        ),
        std::make_unique<interpreter::VariableExpr>(maskVar)
    ));
    
    return verification;
}

std::unique_ptr<interpreter::Statement>
SemanticPreservingMutator::applyLanePermutation(
    const interpreter::Statement* stmt,
    size_t waveOpIndex,
    ProgramState& state,
    FuzzedDataProvider& provider) {
    
    // Check if we can apply this mutation
    if (!tracker.canApplyMutation(stmt, MutationType::LanePermutation, waveOpIndex)) {
        return nullptr;
    }
    
    // For now, just clone the statement and mark the mutation as applied
    // TODO: Implement actual AST transformation
    auto cloned = cloneStatement(stmt);
    if (!cloned) return nullptr;
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::LanePermutation;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Lane permutation (simplified - no AST modification yet)";
    
    // Record the mutation
    tracker.recordMutation(cloned.get(), MutationType::LanePermutation, waveOpIndex, record);
    
    return cloned;
}

std::unique_ptr<interpreter::Statement>
SemanticPreservingMutator::applyParticipantTracking(
    const interpreter::Statement* stmt,
    size_t waveOpIndex,
    ProgramState& state,
    FuzzedDataProvider& provider) {
    
    // Check if we can apply this mutation
    if (!tracker.canApplyMutation(stmt, MutationType::ParticipantTracking, waveOpIndex)) {
        return nullptr;
    }
    
    // For now, just clone the statement and mark the mutation as applied
    // TODO: Implement actual participant tracking
    auto cloned = cloneStatement(stmt);
    if (!cloned) return nullptr;
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::ParticipantTracking;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Participant tracking (simplified - no verification added yet)";
    
    // Record the mutation
    tracker.recordMutation(cloned.get(), MutationType::ParticipantTracking, waveOpIndex, record);
    
    return cloned;
}

MutationType SemanticPreservingMutator::selectMutation(
    const interpreter::Statement* stmt,
    size_t waveOpIndex,
    FuzzedDataProvider& provider) {
    
    // Get available mutations for this wave op
    std::vector<MutationType> available;
    
    if (tracker.canApplyMutation(stmt, MutationType::LanePermutation, waveOpIndex)) {
        available.push_back(MutationType::LanePermutation);
    }
    
    if (tracker.canApplyMutation(stmt, MutationType::ParticipantTracking, waveOpIndex)) {
        available.push_back(MutationType::ParticipantTracking);
    }
    
    if (tracker.canApplyMutation(stmt, MutationType::ValueDuplication, waveOpIndex)) {
        available.push_back(MutationType::ValueDuplication);
    }
    
    if (tracker.canApplyMutation(stmt, MutationType::BroadcastPattern, waveOpIndex)) {
        available.push_back(MutationType::BroadcastPattern);
    }
    
    if (available.empty()) {
        return MutationType::None;
    }
    
    // Choose randomly from available mutations
    size_t choice = provider.ConsumeIntegralInRange<size_t>(0, available.size() - 1);
    return available[choice];
}

std::unique_ptr<interpreter::Statement>
SemanticPreservingMutator::mutateStatement(
    const interpreter::Statement* stmt,
    ProgramState& state,
    FuzzedDataProvider& provider) {
    
    // Find unmutated wave operations
    auto unmutatedOps = tracker.findUnmutatedWaveOps(stmt);
    if (unmutatedOps.empty()) {
        return nullptr;
    }
    
    // Choose a wave op to mutate
    size_t opIndex = unmutatedOps[
        provider.ConsumeIntegralInRange<size_t>(0, unmutatedOps.size() - 1)
    ];
    
    // Select appropriate mutation
    MutationType mutation = selectMutation(stmt, opIndex, provider);
    
    // Apply the mutation
    switch (mutation) {
        case MutationType::LanePermutation:
            return applyLanePermutation(stmt, opIndex, state, provider);
            
        case MutationType::ParticipantTracking:
            return applyParticipantTracking(stmt, opIndex, state, provider);
            
        case MutationType::ValueDuplication:
            // Not implemented yet
            return nullptr;
            
        case MutationType::BroadcastPattern:
            // Not implemented yet
            return nullptr;
            
        default:
            return nullptr;
    }
}

// ExpressionReplacer implementation
std::unique_ptr<interpreter::Expression>
ExpressionReplacer::replaceInExpression(std::unique_ptr<interpreter::Expression> expr) {
    return replaceAtPath(std::move(expr), {});
}

std::unique_ptr<interpreter::Expression>
ExpressionReplacer::replaceAtPath(std::unique_ptr<interpreter::Expression> expr,
                                 std::vector<size_t> currentPath) {
    if (!expr) return nullptr;
    
    // Check if we're at the target location
    if (currentPath == targetPath && !replaced) {
        replaced = true;
        return std::move(replacement);
    }
    
    // TODO: Implement proper AST manipulation with visitor pattern
    // For now, we can't directly access private members
    // This is a limitation that needs to be addressed in the interpreter design
    
    return expr;
}

} // namespace fuzzer
} // namespace minihlsl