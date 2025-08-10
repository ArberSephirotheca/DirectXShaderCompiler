#include "HLSLMutationTracker.h"
#include <algorithm>
#include <cassert>

namespace minihlsl {
namespace fuzzer {

bool MutationTracker::canApplyMutation(const interpreter::Statement* stmt,
                                      MutationType mutation,
                                      size_t waveOpIndex) const {
    auto it = metadataMap.find(stmt);
    if (it == metadataMap.end()) {
        return false;  // Statement not registered
    }
    
    const auto& metadata = it->second;
    
    // Only mutate statements from current round
    if (!isFromCurrentRound(stmt)) {
        return false;
    }
    
    // Check if wave op index is valid
    if (waveOpIndex >= metadata.waveOps.size()) {
        return false;
    }
    
    // Check if this wave op was already mutated
    if (metadata.mutatedWaveOpIndices.count(waveOpIndex) > 0) {
        // Check if the proposed mutation is compatible with existing
        const auto& waveOp = metadata.waveOps[waveOpIndex];
        if (waveOp.isMutated) {
            return areMutationsCompatible(waveOp.appliedMutation, mutation);
        }
    }
    
    // Check global constraints
    if (mutation == MutationType::ParticipantTracking) {
        // Only one participant tracking per wave op globally
        if (globallyMutatedOps.count(metadata.waveOps[waveOpIndex].op) > 0) {
            return false;
        }
    }
    
    return true;
}

void MutationTracker::recordMutation(const interpreter::Statement* stmt,
                                    MutationType mutation,
                                    size_t waveOpIndex,
                                    const MutationRecord& record) {
    auto it = metadataMap.find(stmt);
    if (it == metadataMap.end()) {
        return;  // Statement not registered
    }
    
    auto& metadata = it->second;
    
    // Update mutation tracking
    metadata.appliedMutations.insert(mutation);
    metadata.mutatedWaveOpIndices.insert(waveOpIndex);
    metadata.mutationHistory.push_back(record);
    
    // Update wave op location
    if (waveOpIndex < metadata.waveOps.size()) {
        metadata.waveOps[waveOpIndex].isMutated = true;
        metadata.waveOps[waveOpIndex].appliedMutation = mutation;
        
        // Track globally for certain mutations
        if (mutation == MutationType::ParticipantTracking) {
            globallyMutatedOps.insert(metadata.waveOps[waveOpIndex].op);
        }
    }
}

void MutationTracker::registerStatement(const interpreter::Statement* stmt,
                                       const StatementMetadata& metadata) {
    metadataMap[stmt] = metadata;
}

const StatementMetadata* MutationTracker::getMetadata(const interpreter::Statement* stmt) const {
    auto it = metadataMap.find(stmt);
    return (it != metadataMap.end()) ? &it->second : nullptr;
}

StatementMetadata* MutationTracker::getMutableMetadata(const interpreter::Statement* stmt) {
    auto it = metadataMap.find(stmt);
    return (it != metadataMap.end()) ? &it->second : nullptr;
}

bool MutationTracker::areMutationsCompatible(MutationType existing, MutationType proposed) const {
    // Lane permutation changes input values
    // Participant tracking adds verification
    // These can be combined
    if ((existing == MutationType::LanePermutation && 
         proposed == MutationType::ParticipantTracking) ||
        (existing == MutationType::ParticipantTracking && 
         proposed == MutationType::LanePermutation)) {
        return true;
    }
    
    // Two lane permutations would conflict
    if (existing == MutationType::LanePermutation && 
        proposed == MutationType::LanePermutation) {
        return false;
    }
    
    // Value duplication and broadcast patterns conflict with lane permutation
    if ((existing == MutationType::LanePermutation || existing == MutationType::ValueDuplication ||
         existing == MutationType::BroadcastPattern) &&
        (proposed == MutationType::LanePermutation || proposed == MutationType::ValueDuplication ||
         proposed == MutationType::BroadcastPattern)) {
        return false;
    }
    
    // Participant tracking is generally compatible with value modifications
    if (existing == MutationType::ParticipantTracking || 
        proposed == MutationType::ParticipantTracking) {
        return true;
    }
    
    return false;
}

std::vector<size_t> MutationTracker::findUnmutatedWaveOps(const interpreter::Statement* stmt) const {
    std::vector<size_t> unmutated;
    
    auto it = metadataMap.find(stmt);
    if (it == metadataMap.end()) {
        return unmutated;
    }
    
    const auto& metadata = it->second;
    for (size_t i = 0; i < metadata.waveOps.size(); ++i) {
        if (!metadata.waveOps[i].isMutated) {
            unmutated.push_back(i);
        }
    }
    
    return unmutated;
}

bool MutationTracker::isFromCurrentRound(const interpreter::Statement* stmt) const {
    auto it = metadataMap.find(stmt);
    if (it == metadataMap.end()) {
        return false;
    }
    
    return it->second.generationRound == currentRound;
}

// Helper function to find wave operations in expressions
std::vector<WaveOpLocation> findAllWaveOps(const interpreter::Expression* expr,
                                          std::vector<size_t> currentPath) {
    std::vector<WaveOpLocation> locations;
    
    if (!expr) return locations;
    
    // Check if this expression is a wave operation
    if (auto waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(expr)) {
        locations.emplace_back(waveOp, currentPath);
    }
    
    // Recursively search in child expressions
    if (auto binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
        auto leftPath = currentPath;
        leftPath.push_back(0);
        auto leftOps = findAllWaveOps(binOp->left.get(), leftPath);
        locations.insert(locations.end(), leftOps.begin(), leftOps.end());
        
        auto rightPath = currentPath;
        rightPath.push_back(1);
        auto rightOps = findAllWaveOps(binOp->right.get(), rightPath);
        locations.insert(locations.end(), rightOps.begin(), rightOps.end());
    }
    else if (auto unaryOp = dynamic_cast<const interpreter::UnaryOpExpr*>(expr)) {
        auto childPath = currentPath;
        childPath.push_back(0);
        auto childOps = findAllWaveOps(unaryOp->expr.get(), childPath);
        locations.insert(locations.end(), childOps.begin(), childOps.end());
    }
    else if (auto waveRead = dynamic_cast<const interpreter::WaveReadLaneAt*>(expr)) {
        auto valuePath = currentPath;
        valuePath.push_back(0);
        auto valueOps = findAllWaveOps(waveRead->value.get(), valuePath);
        locations.insert(locations.end(), valueOps.begin(), valueOps.end());
        
        auto lanePath = currentPath;
        lanePath.push_back(1);
        auto laneOps = findAllWaveOps(waveRead->lane.get(), lanePath);
        locations.insert(locations.end(), laneOps.begin(), laneOps.end());
    }
    
    return locations;
}

// Helper function to find wave operations in statements
std::vector<WaveOpLocation> findAllWaveOps(const interpreter::Statement* stmt) {
    std::vector<WaveOpLocation> locations;
    
    if (!stmt) return locations;
    
    // Check different statement types
    if (auto varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
        if (varDecl->initExpr) {
            locations = findAllWaveOps(varDecl->initExpr.get());
        }
    }
    else if (auto assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
        locations = findAllWaveOps(assign->expr.get());
    }
    else if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
        // Search in condition
        auto condOps = findAllWaveOps(ifStmt->condition.get());
        locations.insert(locations.end(), condOps.begin(), condOps.end());
        
        // Search in then block
        for (const auto& thenStmt : ifStmt->thenStmt) {
            auto thenOps = findAllWaveOps(thenStmt.get());
            locations.insert(locations.end(), thenOps.begin(), thenOps.end());
        }
        
        // Search in else block
        for (const auto& elseStmt : ifStmt->elseStmt) {
            auto elseOps = findAllWaveOps(elseStmt.get());
            locations.insert(locations.end(), elseOps.begin(), elseOps.end());
        }
    }
    else if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
        // Search in condition
        if (forStmt->condition) {
            auto condOps = findAllWaveOps(forStmt->condition.get());
            locations.insert(locations.end(), condOps.begin(), condOps.end());
        }
        
        // Search in body
        for (const auto& bodyStmt : forStmt->body) {
            auto bodyOps = findAllWaveOps(bodyStmt.get());
            locations.insert(locations.end(), bodyOps.begin(), bodyOps.end());
        }
    }
    else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
        // Search in condition
        auto condOps = findAllWaveOps(whileStmt->condition.get());
        locations.insert(locations.end(), condOps.begin(), condOps.end());
        
        // Search in body
        for (const auto& bodyStmt : whileStmt->body) {
            auto bodyOps = findAllWaveOps(bodyStmt.get());
            locations.insert(locations.end(), bodyOps.begin(), bodyOps.end());
        }
    }
    
    return locations;
}

// Clone expression helper
std::unique_ptr<interpreter::Expression> cloneExpression(const interpreter::Expression* expr) {
    if (!expr) return nullptr;
    
    // Handle different expression types
    if (auto lit = dynamic_cast<const interpreter::LiteralExpr*>(expr)) {
        return std::make_unique<interpreter::LiteralExpr>(lit->value);
    }
    else if (auto var = dynamic_cast<const interpreter::VariableExpr*>(expr)) {
        return std::make_unique<interpreter::VariableExpr>(var->name);
    }
    else if (auto binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
        return std::make_unique<interpreter::BinaryOpExpr>(
            cloneExpression(binOp->left.get()),
            cloneExpression(binOp->right.get()),
            binOp->op
        );
    }
    else if (auto unaryOp = dynamic_cast<const interpreter::UnaryOpExpr*>(expr)) {
        return std::make_unique<interpreter::UnaryOpExpr>(
            cloneExpression(unaryOp->expr.get()),
            unaryOp->op
        );
    }
    else if (auto waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(expr)) {
        return std::make_unique<interpreter::WaveActiveOp>(
            waveOp->opType,
            cloneExpression(waveOp->expr.get())
        );
    }
    else if (auto laneIndex = dynamic_cast<const interpreter::LaneIndexExpr*>(expr)) {
        return std::make_unique<interpreter::LaneIndexExpr>();
    }
    else if (auto waveRead = dynamic_cast<const interpreter::WaveReadLaneAt*>(expr)) {
        return std::make_unique<interpreter::WaveReadLaneAt>(
            cloneExpression(waveRead->value.get()),
            cloneExpression(waveRead->lane.get())
        );
    }
    else if (auto threadId = dynamic_cast<const interpreter::DispatchThreadIdExpr*>(expr)) {
        return std::make_unique<interpreter::DispatchThreadIdExpr>(threadId->axis);
    }
    else if (auto bufferLoad = dynamic_cast<const interpreter::BufferLoadExpr*>(expr)) {
        return std::make_unique<interpreter::BufferLoadExpr>(
            bufferLoad->bufferName,
            cloneExpression(bufferLoad->index.get())
        );
    }
    else if (auto waveBallot = dynamic_cast<const interpreter::WaveActiveBallot*>(expr)) {
        return std::make_unique<interpreter::WaveActiveBallot>(
            cloneExpression(waveBallot->condition.get())
        );
    }
    
    // If we don't recognize the type, return nullptr
    return nullptr;
}

// Clone statement helper
std::unique_ptr<interpreter::Statement> cloneStatement(const interpreter::Statement* stmt) {
    if (!stmt) return nullptr;
    
    // Handle different statement types
    if (auto varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
        return std::make_unique<interpreter::VarDeclStmt>(
            varDecl->varName,
            varDecl->type,
            cloneExpression(varDecl->initExpr.get())
        );
    }
    else if (auto assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
        return std::make_unique<interpreter::AssignStmt>(
            assign->varName,
            cloneExpression(assign->expr.get())
        );
    }
    else if (auto bufferStore = dynamic_cast<const interpreter::BufferStoreStmt*>(stmt)) {
        return std::make_unique<interpreter::BufferStoreStmt>(
            bufferStore->bufferName,
            cloneExpression(bufferStore->index.get()),
            cloneExpression(bufferStore->value.get())
        );
    }
    else if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
        std::vector<std::unique_ptr<interpreter::Statement>> thenClone;
        for (const auto& s : ifStmt->thenStmt) {
            thenClone.push_back(cloneStatement(s.get()));
        }
        
        std::vector<std::unique_ptr<interpreter::Statement>> elseClone;
        for (const auto& s : ifStmt->elseStmt) {
            elseClone.push_back(cloneStatement(s.get()));
        }
        
        return std::make_unique<interpreter::IfStmt>(
            cloneExpression(ifStmt->condition.get()),
            std::move(thenClone),
            std::move(elseClone)
        );
    }
    else if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
        std::vector<std::unique_ptr<interpreter::Statement>> bodyClone;
        for (const auto& s : forStmt->body) {
            bodyClone.push_back(cloneStatement(s.get()));
        }
        
        return std::make_unique<interpreter::ForStmt>(
            forStmt->var,
            cloneExpression(forStmt->init.get()),
            cloneExpression(forStmt->condition.get()),
            cloneExpression(forStmt->increment.get()),
            std::move(bodyClone)
        );
    }
    else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
        std::vector<std::unique_ptr<interpreter::Statement>> bodyClone;
        for (const auto& s : whileStmt->body) {
            bodyClone.push_back(cloneStatement(s.get()));
        }
        
        return std::make_unique<interpreter::WhileStmt>(
            cloneExpression(whileStmt->condition.get()),
            std::move(bodyClone)
        );
    }
    else if (auto breakStmt = dynamic_cast<const interpreter::BreakStmt*>(stmt)) {
        return std::make_unique<interpreter::BreakStmt>();
    }
    else if (auto continueStmt = dynamic_cast<const interpreter::ContinueStmt*>(stmt)) {
        return std::make_unique<interpreter::ContinueStmt>();
    }
    
    // If we don't recognize the type, return nullptr
    return nullptr;
}

} // namespace fuzzer
} // namespace minihlsl