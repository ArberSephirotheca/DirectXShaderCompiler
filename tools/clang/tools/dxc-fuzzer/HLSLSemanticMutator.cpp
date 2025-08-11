#include "HLSLSemanticMutator.h"
#include <sstream>
#include <cassert>

namespace minihlsl {
namespace fuzzer {

// Helper function to get permutation pattern description
static std::string getPermutationDescription(SemanticPreservingMutator::PermutationPattern pattern) {
    switch (pattern) {
        case SemanticPreservingMutator::PermutationPattern::Rotate:
            return "Rotate";
        case SemanticPreservingMutator::PermutationPattern::Reverse:
            return "Reverse";
        case SemanticPreservingMutator::PermutationPattern::EvenOddSwap:
            return "EvenOddSwap";
        case SemanticPreservingMutator::PermutationPattern::Butterfly:
            return "Butterfly";
        case SemanticPreservingMutator::PermutationPattern::Broadcast:
            return "Broadcast";
        case SemanticPreservingMutator::PermutationPattern::Custom:
            return "Custom";
        default:
            return "Unknown";
    }
}

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
    
    std::cerr << "DEBUG: applyLanePermutation called for waveOpIndex=" << waveOpIndex << "\n";
    
    // Check if we can apply this mutation
    if (!tracker.canApplyMutation(stmt, MutationType::LanePermutation, waveOpIndex)) {
        std::cerr << "DEBUG: applyLanePermutation - cannot apply mutation\n";
        return nullptr;
    }
    
    // Get metadata to find participant count
    auto metadata = tracker.getMetadata(stmt);
    if (!metadata || waveOpIndex >= metadata->waveOps.size()) {
        return nullptr;
    }
    
    // Determine participant count (use wave size as approximation)
    uint32_t participantCount = state.program.waveSizePreferred > 0 ? 
                                state.program.waveSizePreferred : 32;
    
    // Choose permutation pattern
    auto pattern = static_cast<PermutationPattern>(
        provider.ConsumeIntegralInRange<int>(0, 5));
    
    // Generate permutation expression
    auto permExpr = generatePermutationExpr(pattern, participantCount, provider);
    
    // Create replacement function that wraps wave operations
    std::function<std::unique_ptr<interpreter::Expression>(
        std::unique_ptr<interpreter::Expression>)> replacementFunc = 
        [&permExpr](std::unique_ptr<interpreter::Expression> waveOp) -> std::unique_ptr<interpreter::Expression> {
        // For wave operations, wrap the input with WaveReadLaneAt
        if (auto wave = dynamic_cast<interpreter::WaveActiveOp*>(waveOp.get())) {
            // Get the input expression
            auto input = wave->getInput();
            if (!input) return waveOp;
            
            // Clone the input
            auto inputClone = input->clone();
            
            // Wrap with WaveReadLaneAt using our permutation
            auto permutedInput = std::make_unique<interpreter::WaveReadLaneAt>(
                std::move(inputClone),
                permExpr->clone(),
                input->getType()
            );
            
            // Create new wave operation with permuted input
            return std::make_unique<interpreter::WaveActiveOp>(
                std::move(permutedInput),
                wave->getOpType()
            );
        }
        return waveOp;
    };
    
    // Use WaveOpReplacer to modify the statement
    WaveOpReplacer replacer(waveOpIndex, replacementFunc);
    auto mutatedStmt = replacer.replaceInStatement(cloneStatement(stmt));
    
    std::cerr << "DEBUG: applyLanePermutation - replacer.wasReplaced()=" << replacer.wasReplaced() << "\n";
    
    if (!mutatedStmt || !replacer.wasReplaced()) {
        std::cerr << "DEBUG: applyLanePermutation - mutation failed\n";
        return nullptr;
    }
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::LanePermutation;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Lane permutation: " + getPermutationDescription(pattern);
    
    // Record the mutation
    tracker.recordMutation(mutatedStmt.get(), MutationType::LanePermutation, waveOpIndex, record);
    
    return mutatedStmt;
}

std::unique_ptr<interpreter::Statement>
SemanticPreservingMutator::applyParticipantTracking(
    const interpreter::Statement* stmt,
    size_t waveOpIndex,
    ProgramState& state,
    FuzzedDataProvider& provider) {
    
    std::cerr << "DEBUG: applyParticipantTracking called\n";
    
    // Check if we can apply this mutation
    if (!tracker.canApplyMutation(stmt, MutationType::ParticipantTracking, waveOpIndex)) {
        std::cerr << "DEBUG: applyParticipantTracking - cannot apply mutation\n";
        return nullptr;
    }
    
    // Get metadata
    auto metadata = tracker.getMetadata(stmt);
    if (!metadata || waveOpIndex >= metadata->waveOps.size()) {
        return nullptr;
    }
    
    // Clone the original statement
    auto clonedStmt = cloneStatement(stmt);
    if (!clonedStmt) {
        return nullptr;
    }
    
    // Create tracking statements that will wrap the original statement
    std::vector<std::unique_ptr<interpreter::Statement>> trackingStatements;
    
    // 1. Count active participants before wave op: _participantCount = WaveActiveSum(1)
    auto oneExpr = std::make_unique<interpreter::LiteralExpr>(1);
    auto sumOp = std::make_unique<interpreter::WaveActiveOp>(
        std::move(oneExpr), interpreter::WaveActiveOp::Sum);
    
    std::string countVarName = "_participant_count_" + std::to_string(waveOpIndex);
    trackingStatements.push_back(std::make_unique<interpreter::VarDeclStmt>(
        countVarName, interpreter::HLSLType::Uint, std::move(sumOp)));
    
    // 2. Add the original statement (with wave operation)
    trackingStatements.push_back(std::move(clonedStmt));
    
    // 3. Count participants after wave op: _participantCountPost = WaveActiveSum(1)
    auto oneExpr2 = std::make_unique<interpreter::LiteralExpr>(1);
    auto sumOp2 = std::make_unique<interpreter::WaveActiveOp>(
        std::move(oneExpr2), interpreter::WaveActiveOp::Sum);
    
    std::string countVarNamePost = "_participant_count_post_" + std::to_string(waveOpIndex);
    trackingStatements.push_back(std::make_unique<interpreter::VarDeclStmt>(
        countVarNamePost, interpreter::HLSLType::Uint, std::move(sumOp2)));
    
    // 4. Check if counts match: _counts_match = (pre == post)
    auto preCountRef = std::make_unique<interpreter::VariableExpr>(countVarName);
    auto postCountRef = std::make_unique<interpreter::VariableExpr>(countVarNamePost);
    auto compare = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(preCountRef), std::move(postCountRef), interpreter::BinaryOpExpr::Eq);
    
    std::string matchVarName = "_counts_match_" + std::to_string(waveOpIndex);
    trackingStatements.push_back(std::make_unique<interpreter::VarDeclStmt>(
        matchVarName, interpreter::HLSLType::Bool, std::move(compare)));
    
    // 5. Store verification result to buffer: _participant_check_sum[tid.x] += matchResult
    // Convert bool to uint
    auto matchRef = std::make_unique<interpreter::VariableExpr>(matchVarName);
    auto matchAsUint = std::make_unique<interpreter::ConditionalExpr>(
        std::move(matchRef),
        std::make_unique<interpreter::LiteralExpr>(1),
        std::make_unique<interpreter::LiteralExpr>(0)
    );
    
    // Get thread ID component
    auto tidX = std::make_unique<interpreter::DispatchThreadIdExpr>(0); // component 0 = x
    
    // Read current buffer value
    auto bufferAccess = std::make_unique<interpreter::ArrayAccessExpr>(
        "_participant_check_sum",
        tidX->clone(),
        interpreter::HLSLType::Uint
    );
    
    // Add to current value
    auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(bufferAccess), std::move(matchAsUint),
        interpreter::BinaryOpExpr::Add
    );
    
    // Store back to buffer
    trackingStatements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
        "_participant_check_sum", std::move(tidX), std::move(addExpr)));
    
    // Wrap all statements in if(true) to create a single statement
    auto trueCond = std::make_unique<interpreter::LiteralExpr>(true);
    auto wrappedStmt = std::make_unique<interpreter::IfStmt>(
        std::move(trueCond), 
        std::move(trackingStatements),
        std::vector<std::unique_ptr<interpreter::Statement>>{}
    );
    
    // Create mutation record
    MutationRecord record;
    record.type = MutationType::ParticipantTracking;
    record.waveOpIndex = waveOpIndex;
    record.round = tracker.getCurrentRound();
    record.description = "Participant tracking with WaveActiveSum verification";
    
    // Record the mutation
    tracker.recordMutation(wrappedStmt.get(), MutationType::ParticipantTracking, waveOpIndex, record);
    
    return wrappedStmt;
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
    
    // TODO: Enable these once implemented
    // if (tracker.canApplyMutation(stmt, MutationType::ValueDuplication, waveOpIndex)) {
    //     available.push_back(MutationType::ValueDuplication);
    // }
    
    // if (tracker.canApplyMutation(stmt, MutationType::BroadcastPattern, waveOpIndex)) {
    //     available.push_back(MutationType::BroadcastPattern);
    // }
    
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
    std::cerr << "DEBUG: mutateStatement - found " << unmutatedOps.size() << " unmutated wave ops\n";
    if (unmutatedOps.empty()) {
        return nullptr;
    }
    
    // Choose a wave op to mutate
    size_t opIndex = unmutatedOps[
        provider.ConsumeIntegralInRange<size_t>(0, unmutatedOps.size() - 1)
    ];
    
    // Select appropriate mutation
    MutationType mutation = selectMutation(stmt, opIndex, provider);
    std::cerr << "DEBUG: mutateStatement - selected mutation type: " << static_cast<int>(mutation) << "\n";
    
    // Apply the mutation
    switch (mutation) {
        case MutationType::LanePermutation:
            std::cerr << "DEBUG: mutateStatement - applying lane permutation\n";
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

// WaveOpReplacer implementation
std::unique_ptr<interpreter::Expression>
WaveOpReplacer::replaceInExpression(std::unique_ptr<interpreter::Expression> expr) {
    if (!expr) return nullptr;
    
    // Check if this is a wave operation
    if (auto waveOp = dynamic_cast<interpreter::WaveActiveOp*>(expr.get())) {
        if (currentWaveOpCount == targetWaveOpIndex && !replaced) {
            replaced = true;
            // Apply the replacement function
            return replacementFunc(std::move(expr));
        }
        currentWaveOpCount++;
        // Still need to clone even if not replacing
        return expr->clone();
    }
    
    // Recursively process sub-expressions
    if (auto binOp = dynamic_cast<interpreter::BinaryOpExpr*>(expr.get())) {
        auto left = binOp->getLeft();
        auto right = binOp->getRight();
        
        auto newLeft = replaceInExpression(left ? left->clone() : nullptr);
        auto newRight = replaceInExpression(right ? right->clone() : nullptr);
        
        if (replaced) {
            return std::make_unique<interpreter::BinaryOpExpr>(
                std::move(newLeft), std::move(newRight), binOp->getOp());
        }
    }
    else if (auto unaryOp = dynamic_cast<interpreter::UnaryOpExpr*>(expr.get())) {
        auto inner = unaryOp->getExpr();
        auto newInner = replaceInExpression(inner ? inner->clone() : nullptr);
        
        if (replaced) {
            return std::make_unique<interpreter::UnaryOpExpr>(
                std::move(newInner), unaryOp->getOp());
        }
    }
    else if (auto waveRead = dynamic_cast<interpreter::WaveReadLaneAt*>(expr.get())) {
        auto value = waveRead->getValue();
        auto lane = waveRead->getLaneIndex();
        
        auto newValue = replaceInExpression(value ? value->clone() : nullptr);
        auto newLane = replaceInExpression(lane ? lane->clone() : nullptr);
        
        if (replaced) {
            return std::make_unique<interpreter::WaveReadLaneAt>(
                std::move(newValue), std::move(newLane), waveRead->getType());
        }
    }
    
    // For other expression types, return clone
    return expr->clone();
}

std::unique_ptr<interpreter::Statement>
WaveOpReplacer::replaceInStatement(std::unique_ptr<interpreter::Statement> stmt) {
    if (!stmt) return nullptr;
    
    // Handle different statement types
    if (auto varDecl = dynamic_cast<interpreter::VarDeclStmt*>(stmt.get())) {
        auto init = varDecl->getInit();
        auto newInit = replaceInExpression(init ? init->clone() : nullptr);
        
        if (replaced) {
            return std::make_unique<interpreter::VarDeclStmt>(
                varDecl->getName(), varDecl->getType(), std::move(newInit));
        }
    }
    else if (auto assign = dynamic_cast<interpreter::AssignStmt*>(stmt.get())) {
        auto expr = assign->getExpression();
        auto newExpr = replaceInExpression(expr ? expr->clone() : nullptr);
        
        if (replaced) {
            return std::make_unique<interpreter::AssignStmt>(
                assign->getName(), std::move(newExpr));
        }
    }
    else if (auto arrayAssign = dynamic_cast<interpreter::ArrayAssignStmt*>(stmt.get())) {
        auto value = arrayAssign->getValueExpr();
        auto newValue = replaceInExpression(value ? value->clone() : nullptr);
        
        if (replaced) {
            auto index = arrayAssign->getIndexExpr();
            return std::make_unique<interpreter::ArrayAssignStmt>(
                arrayAssign->getArrayName(),
                index ? index->clone() : nullptr,
                std::move(newValue));
        }
    }
    else if (auto ifStmt = dynamic_cast<interpreter::IfStmt*>(stmt.get())) {
        // Process then body
        std::vector<std::unique_ptr<interpreter::Statement>> newThenBody;
        bool modifiedThen = false;
        for (const auto& s : ifStmt->getThenBlock()) {
            auto newStmt = replaceInStatement(s->clone());
            if (replaced && !modifiedThen) {
                modifiedThen = true;
            }
            newThenBody.push_back(std::move(newStmt));
        }
        
        // Process else body
        std::vector<std::unique_ptr<interpreter::Statement>> newElseBody;
        bool modifiedElse = false;
        for (const auto& s : ifStmt->getElseBlock()) {
            auto newStmt = replaceInStatement(s->clone());
            if (replaced && !modifiedElse) {
                modifiedElse = true;
            }
            newElseBody.push_back(std::move(newStmt));
        }
        
        if (modifiedThen || modifiedElse) {
            return std::make_unique<interpreter::IfStmt>(
                ifStmt->getCondition()->clone(),
                std::move(newThenBody),
                std::move(newElseBody)
            );
        }
    }
    else if (auto forStmt = dynamic_cast<interpreter::ForStmt*>(stmt.get())) {
        // Process for loop body
        std::vector<std::unique_ptr<interpreter::Statement>> newBody;
        bool modified = false;
        for (const auto& s : forStmt->getBody()) {
            auto newStmt = replaceInStatement(s->clone());
            if (replaced && !modified) {
                modified = true;
            }
            newBody.push_back(std::move(newStmt));
        }
        
        if (modified) {
            return std::make_unique<interpreter::ForStmt>(
                forStmt->getLoopVar(),
                forStmt->getInit()->clone(),
                forStmt->getCondition()->clone(),
                forStmt->getIncrement()->clone(),
                std::move(newBody)
            );
        }
    }
    else if (auto whileStmt = dynamic_cast<interpreter::WhileStmt*>(stmt.get())) {
        // Process while loop body
        std::vector<std::unique_ptr<interpreter::Statement>> newBody;
        bool modified = false;
        for (const auto& s : whileStmt->getBody()) {
            auto newStmt = replaceInStatement(s->clone());
            if (replaced && !modified) {
                modified = true;
            }
            newBody.push_back(std::move(newStmt));
        }
        
        if (modified) {
            return std::make_unique<interpreter::WhileStmt>(
                whileStmt->getCondition()->clone(),
                std::move(newBody)
            );
        }
    }
    
    return stmt->clone();
}

} // namespace fuzzer
} // namespace minihlsl