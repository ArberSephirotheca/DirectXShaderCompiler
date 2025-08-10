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
                interpreter::BinaryOpExpr::Add
            );
            
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(sum),
                std::move(countExpr),
                interpreter::BinaryOpExpr::Mod
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
                interpreter::BinaryOpExpr::Sub
            );
        }
        
        case PermutationPattern::EvenOddSwap: {
            // lane ^ 1 (swaps bit 0)
            auto one = std::make_unique<interpreter::LiteralExpr>(1);
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(laneIndex),
                std::move(one),
                interpreter::BinaryOpExpr::Xor
            );
        }
        
        case PermutationPattern::Butterfly: {
            // lane ^ (1 << stage)
            uint32_t stage = provider.ConsumeIntegralInRange<uint32_t>(0, 4);
            auto mask = std::make_unique<interpreter::LiteralExpr>(1 << stage);
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(laneIndex),
                std::move(mask),
                interpreter::BinaryOpExpr::Xor
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
    
    // Get metadata to find the wave op location
    auto metadata = tracker.getMutableMetadata(stmt);
    if (!metadata || waveOpIndex >= metadata->waveOps.size()) {
        return false;
    }
    
    const auto& waveOpLoc = metadata->waveOps[waveOpIndex];
    
    // Use ExpressionReplacer to replace at the correct path
    ExpressionReplacer replacer(waveOpLoc.path, std::move(replacement));
    
    // Apply replacement based on statement type
    if (auto varDecl = dynamic_cast<interpreter::VarDeclStmt*>(stmt)) {
        if (varDecl->initExpr) {
            varDecl->initExpr = replacer.replaceInExpression(std::move(varDecl->initExpr));
            return replacer.wasReplaced();
        }
    }
    else if (auto assign = dynamic_cast<interpreter::AssignStmt*>(stmt)) {
        assign->expr = replacer.replaceInExpression(std::move(assign->expr));
        return replacer.wasReplaced();
    }
    
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
        std::make_unique<interpreter::WaveActiveBallot>(
            std::make_unique<interpreter::LiteralExpr>(1)  // true
        )
    ));
    
    // Store to buffer for verification
    verification.push_back(std::make_unique<interpreter::BufferStoreStmt>(
        "_participant_check_sum",
        std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::DispatchThreadIdExpr>(0),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(waveOpIndex * 2 + (phase == "pre" ? 0 : 1))),
            interpreter::BinaryOpExpr::Add
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
    
    // Clone the statement
    auto cloned = cloneStatement(stmt);
    if (!cloned) return nullptr;
    
    // Get metadata
    auto metadata = tracker.getMetadata(stmt);
    if (!metadata || waveOpIndex >= metadata->waveOps.size()) {
        return nullptr;
    }
    
    // Choose permutation pattern
    auto pattern = static_cast<PermutationPattern>(
        provider.ConsumeIntegralInRange<int>(0, 5)
    );
    
    // Estimate participant count (this is simplified - in reality would analyze the condition)
    uint32_t participantCount = provider.ConsumeIntegralInRange<uint32_t>(2, 32);
    
    // Generate permutation expression
    auto permExpr = generatePermutationExpr(pattern, participantCount, provider);
    
    // Find the wave operation and wrap its input
    // This is simplified - in reality we'd need to traverse the AST
    // For now, assume we can find and modify it
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::LanePermutation;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Applied " + std::to_string(static_cast<int>(pattern)) + " permutation";
    
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
    
    // Create a vector to hold the compound statement
    std::vector<std::unique_ptr<interpreter::Statement>> statements;
    
    // Add pre-verification
    auto preVerify = generateParticipantVerification("pre", waveOpIndex, state);
    for (auto& s : preVerify) {
        statements.push_back(std::move(s));
    }
    
    // Add the original statement (cloned)
    statements.push_back(cloneStatement(stmt));
    
    // Add post-verification
    auto postVerify = generateParticipantVerification("post", waveOpIndex, state);
    for (auto& s : postVerify) {
        statements.push_back(std::move(s));
    }
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::ParticipantTracking;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Added participant tracking verification";
    
    // Record the mutation on the cloned statement
    tracker.recordMutation(statements[preVerify.size()].get(), 
                          MutationType::ParticipantTracking, waveOpIndex, record);
    
    // Return the first statement (caller will need to handle multiple statements)
    // In a real implementation, we'd return a compound statement
    return std::move(statements[0]);
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
    
    // Continue traversing based on expression type
    if (auto binOp = dynamic_cast<interpreter::BinaryOpExpr*>(expr.get())) {
        auto leftPath = currentPath;
        leftPath.push_back(0);
        binOp->left = replaceAtPath(std::move(binOp->left), leftPath);
        
        auto rightPath = currentPath;
        rightPath.push_back(1);
        binOp->right = replaceAtPath(std::move(binOp->right), rightPath);
    }
    else if (auto unaryOp = dynamic_cast<interpreter::UnaryOpExpr*>(expr.get())) {
        auto childPath = currentPath;
        childPath.push_back(0);
        unaryOp->expr = replaceAtPath(std::move(unaryOp->expr), childPath);
    }
    else if (auto waveOp = dynamic_cast<interpreter::WaveActiveOp*>(expr.get())) {
        auto childPath = currentPath;
        childPath.push_back(0);
        waveOp->expr = replaceAtPath(std::move(waveOp->expr), childPath);
    }
    
    return expr;
}

} // namespace fuzzer
} // namespace minihlsl