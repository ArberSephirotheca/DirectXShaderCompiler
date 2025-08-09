#ifndef HLSL_PARTICIPANT_PATTERNS_H
#define HLSL_PARTICIPANT_PATTERNS_H

#include "MiniHLSLInterpreter.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <memory>
#include <set>
#include <string>

namespace minihlsl {
namespace fuzzer {

// Base class for all participant patterns
class ParticipantPattern {
public:
    virtual ~ParticipantPattern() = default;
    
    // Generate the condition expression that creates this participant pattern
    virtual std::unique_ptr<interpreter::Expression> 
        generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) = 0;
    
    // Get the expected set of participating lanes for verification
    virtual std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) = 0;
    
    // Get a description of this pattern
    virtual std::string getDescription() const = 0;
};

// Pattern 1: Single lane participation
class SingleLanePattern : public ParticipantPattern {
private:
    uint32_t targetLane;
    
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Pattern 2: Sparse non-contiguous lanes
class SparseNonContiguousPattern : public ParticipantPattern {
private:
    std::set<uint32_t> selectedLanes;
    
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Pattern 3: Contiguous range (first N or last N lanes)
class ContiguousRangePattern : public ParticipantPattern {
private:
    uint32_t count;
    bool fromStart; // true = first N lanes, false = last N lanes
    
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Pattern 4: Disjoint sets (two separate groups)
class DisjointSetsPattern : public ParticipantPattern {
private:
    uint32_t firstGroupEnd;
    uint32_t secondGroupStart;
    
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Pattern 5: Parity pattern (even/odd lanes)
class ParityPattern : public ParticipantPattern {
private:
    bool evenLanes; // true = even lanes, false = odd lanes
    
public:
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Pattern 6: Ensure non-empty wrapper
// This pattern wraps another pattern and ensures at least one lane participates
class EnsureNonEmptyPattern : public ParticipantPattern {
private:
    std::unique_ptr<ParticipantPattern> innerPattern;
    uint32_t guaranteedLane;
    
public:
    explicit EnsureNonEmptyPattern(std::unique_ptr<ParticipantPattern> inner);
    
    std::unique_ptr<interpreter::Expression> 
    generateCondition(uint32_t waveSize, FuzzedDataProvider& provider) override;
    
    std::set<uint32_t> getExpectedParticipants(uint32_t waveSize) override;
    
    std::string getDescription() const override;
};

// Factory function to create random participant patterns
std::unique_ptr<ParticipantPattern> createRandomPattern(FuzzedDataProvider& provider);

} // namespace fuzzer
} // namespace minihlsl

#endif // HLSL_PARTICIPANT_PATTERNS_H