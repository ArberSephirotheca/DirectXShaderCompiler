#include "HLSLParticipantPatterns.h"

namespace minihlsl {
namespace fuzzer {

// SingleLanePattern implementation
std::unique_ptr<interpreter::Expression> 
SingleLanePattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    targetLane = provider.ConsumeIntegralInRange<uint32_t>(0, waveSize - 1);
    
    // Generate: WaveGetLaneIndex() == targetLane
    return std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(targetLane)),
        interpreter::BinaryOpExpr::Eq
    );
}

std::set<uint32_t> SingleLanePattern::getExpectedParticipants(uint32_t waveSize) {
    return {targetLane};
}

std::string SingleLanePattern::getDescription() const {
    return "Single lane " + std::to_string(targetLane);
}

// SparseNonContiguousPattern implementation
std::unique_ptr<interpreter::Expression> 
SparseNonContiguousPattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    selectedLanes.clear();
    
    // Select 2-4 non-contiguous lanes
    uint32_t numLanes = provider.ConsumeIntegralInRange<uint32_t>(2, std::min(4u, waveSize/2));
    
    // Ensure non-contiguous by selecting from different regions
    uint32_t regionSize = waveSize / numLanes;
    for (uint32_t i = 0; i < numLanes; ++i) {
        uint32_t regionStart = i * regionSize;
        uint32_t regionEnd = std::min(regionStart + regionSize, waveSize);
        if (regionStart < regionEnd) {
            uint32_t lane = provider.ConsumeIntegralInRange<uint32_t>(regionStart, regionEnd - 1);
            selectedLanes.insert(lane);
        }
    }
    
    // Build condition: WaveGetLaneIndex() == lane0 || WaveGetLaneIndex() == lane1 || ...
    std::unique_ptr<interpreter::Expression> condition = nullptr;
    
    for (uint32_t lane : selectedLanes) {
        auto laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(lane)),
            interpreter::BinaryOpExpr::Eq
        );
        
        if (!condition) {
            condition = std::move(laneCheck);
        } else {
            condition = std::make_unique<interpreter::BinaryOpExpr>(
                std::move(condition),
                std::move(laneCheck),
                interpreter::BinaryOpExpr::Or
            );
        }
    }
    
    return condition;
}

std::set<uint32_t> SparseNonContiguousPattern::getExpectedParticipants(uint32_t waveSize) {
    return selectedLanes;
}

std::string SparseNonContiguousPattern::getDescription() const {
    std::string desc = "Sparse non-contiguous lanes: ";
    bool first = true;
    for (uint32_t lane : selectedLanes) {
        if (!first) desc += ", ";
        desc += std::to_string(lane);
        first = false;
    }
    return desc;
}

// ContiguousRangePattern implementation
std::unique_ptr<interpreter::Expression> 
ContiguousRangePattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    fromStart = provider.ConsumeBool();
    count = provider.ConsumeIntegralInRange<uint32_t>(1, waveSize / 2);
    
    if (fromStart) {
        // Generate: WaveGetLaneIndex() < count
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(count)),
            interpreter::BinaryOpExpr::Lt
        );
    } else {
        // Generate: WaveGetLaneIndex() >= (waveSize - count)
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(waveSize - count)),
            interpreter::BinaryOpExpr::Ge
        );
    }
}

std::set<uint32_t> ContiguousRangePattern::getExpectedParticipants(uint32_t waveSize) {
    std::set<uint32_t> participants;
    if (fromStart) {
        for (uint32_t i = 0; i < count; ++i) {
            participants.insert(i);
        }
    } else {
        for (uint32_t i = waveSize - count; i < waveSize; ++i) {
            participants.insert(i);
        }
    }
    return participants;
}

std::string ContiguousRangePattern::getDescription() const {
    if (fromStart) {
        return "First " + std::to_string(count) + " lanes";
    } else {
        return "Last " + std::to_string(count) + " lanes";
    }
}

// DisjointSetsPattern implementation
std::unique_ptr<interpreter::Expression> 
DisjointSetsPattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    // Create two groups with a gap in between
    firstGroupEnd = provider.ConsumeIntegralInRange<uint32_t>(1, waveSize / 3);
    secondGroupStart = provider.ConsumeIntegralInRange<uint32_t>(2 * waveSize / 3, waveSize - 1);
    
    // Generate: WaveGetLaneIndex() < firstGroupEnd || WaveGetLaneIndex() >= secondGroupStart
    auto firstGroup = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(firstGroupEnd)),
        interpreter::BinaryOpExpr::Lt
    );
    
    auto secondGroup = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(secondGroupStart)),
        interpreter::BinaryOpExpr::Ge
    );
    
    return std::make_unique<interpreter::BinaryOpExpr>(
        std::move(firstGroup),
        std::move(secondGroup),
        interpreter::BinaryOpExpr::Or
    );
}

std::set<uint32_t> DisjointSetsPattern::getExpectedParticipants(uint32_t waveSize) {
    std::set<uint32_t> participants;
    for (uint32_t i = 0; i < firstGroupEnd; ++i) {
        participants.insert(i);
    }
    for (uint32_t i = secondGroupStart; i < waveSize; ++i) {
        participants.insert(i);
    }
    return participants;
}

std::string DisjointSetsPattern::getDescription() const {
    return "Disjoint sets: lanes < " + std::to_string(firstGroupEnd) + 
           " or lanes >= " + std::to_string(secondGroupStart);
}

// ParityPattern implementation
std::unique_ptr<interpreter::Expression> 
ParityPattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    evenLanes = provider.ConsumeBool();
    
    // Generate: (WaveGetLaneIndex() & 1) == 0 (for even) or == 1 (for odd)
    auto andExpr = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(1),
        interpreter::BinaryOpExpr::BitwiseAnd
    );
    
    return std::make_unique<interpreter::BinaryOpExpr>(
        std::move(andExpr),
        std::make_unique<interpreter::LiteralExpr>(evenLanes ? 0 : 1),
        interpreter::BinaryOpExpr::Eq
    );
}

std::set<uint32_t> ParityPattern::getExpectedParticipants(uint32_t waveSize) {
    std::set<uint32_t> participants;
    for (uint32_t i = 0; i < waveSize; ++i) {
        if ((i & 1) == (evenLanes ? 0 : 1)) {
            participants.insert(i);
        }
    }
    return participants;
}

std::string ParityPattern::getDescription() const {
    return evenLanes ? "Even lanes" : "Odd lanes";
}

// EnsureNonEmptyPattern implementation
EnsureNonEmptyPattern::EnsureNonEmptyPattern(std::unique_ptr<ParticipantPattern> inner)
    : innerPattern(std::move(inner)), guaranteedLane(0) {}

std::unique_ptr<interpreter::Expression> 
EnsureNonEmptyPattern::generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) {
    // First generate the inner pattern's condition
    auto innerCondition = innerPattern->generateCondition(waveSize, provider);
    
    // Pick a guaranteed lane
    guaranteedLane = provider.ConsumeIntegralInRange<uint32_t>(0, waveSize - 1);
    
    // Generate: innerCondition || WaveGetLaneIndex() == guaranteedLane
    auto guaranteeCheck = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(guaranteedLane)),
        interpreter::BinaryOpExpr::Eq
    );
    
    return std::make_unique<interpreter::BinaryOpExpr>(
        std::move(innerCondition),
        std::move(guaranteeCheck),
        interpreter::BinaryOpExpr::Or
    );
}

std::set<uint32_t> EnsureNonEmptyPattern::getExpectedParticipants(uint32_t waveSize) {
    auto participants = innerPattern->getExpectedParticipants(waveSize);
    participants.insert(guaranteedLane);
    return participants;
}

std::string EnsureNonEmptyPattern::getDescription() const {
    return innerPattern->getDescription() + " (guaranteed lane " + std::to_string(guaranteedLane) + ")";
}

// Factory function
std::unique_ptr<ParticipantPattern> createRandomPattern(FuzzedDataProvider& provider) {
    enum PatternChoice {
        SINGLE, SPARSE, CONTIGUOUS, DISJOINT, PARITY, ENSURE_NON_EMPTY, MAX_PATTERN
    };
    
    auto choice = static_cast<PatternChoice>(
        provider.ConsumeIntegralInRange<int>(0, static_cast<int>(MAX_PATTERN) - 1));
    
    std::unique_ptr<ParticipantPattern> pattern;
    
    switch (choice) {
        case SINGLE:
            pattern = std::make_unique<SingleLanePattern>();
            break;
        case SPARSE:
            pattern = std::make_unique<SparseNonContiguousPattern>();
            break;
        case CONTIGUOUS:
            pattern = std::make_unique<ContiguousRangePattern>();
            break;
        case DISJOINT:
            pattern = std::make_unique<DisjointSetsPattern>();
            break;
        case PARITY:
            pattern = std::make_unique<ParityPattern>();
            break;
        case ENSURE_NON_EMPTY:
            // Create a nested pattern with guarantee
            auto innerPattern = std::make_unique<SparseNonContiguousPattern>();
            pattern = std::make_unique<EnsureNonEmptyPattern>(std::move(innerPattern));
            break;
    }
    
    return pattern;
}

} // namespace fuzzer
} // namespace minihlsl