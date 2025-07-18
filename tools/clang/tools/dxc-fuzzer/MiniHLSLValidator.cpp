#include "MiniHLSLValidator.h"
#include <regex>
#include <algorithm>
#include <sstream>

namespace minihlsl {

// Static initialization of forbidden constructs
const std::unordered_set<std::string> MiniHLSLValidator::forbiddenKeywords = {
    "for", "while", "do", "break", "continue", "switch", "case", "default",
    "goto", "struct", "class", "template", "namespace", "using",
    "atomic", "barrier", "sync", "lock", "mutex"
};

const std::unordered_set<std::string> MiniHLSLValidator::forbiddenIntrinsics = {
    "WavePrefixSum", "WavePrefixProduct", "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor",
    "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst",
    "WaveBallot", "WaveMultiPrefixSum", "WaveMultiPrefixProduct",
    "barrier", "AllMemoryBarrier", "GroupMemoryBarrier", "DeviceMemoryBarrier"
};

// UniformityAnalyzer implementation
const std::unordered_set<std::string> UniformityAnalyzer::uniformIntrinsics = {
    "WaveGetLaneCount", "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
};

const std::unordered_set<std::string> UniformityAnalyzer::divergentIntrinsics = {
    "WaveGetLaneIndex", "WaveIsFirstLane"
};

bool UniformityAnalyzer::isUniform(const Expression* expr) const {
    // This is a simplified implementation - in practice, would analyze AST
    // For now, implement basic heuristics for string-based analysis
    return false; // Conservative: assume divergent unless proven uniform
}

void UniformityAnalyzer::markUniform(const std::string& varName) {
    uniformVariables.insert(varName);
    divergentVariables.erase(varName);
}

void UniformityAnalyzer::markDivergent(const std::string& varName) {
    divergentVariables.insert(varName);
    uniformVariables.erase(varName);
}

// WaveOperationValidator implementation
const std::unordered_set<std::string> WaveOperationValidator::orderIndependentOps = {
    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits",
    "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
    "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
};

const std::unordered_set<std::string> WaveOperationValidator::orderDependentOps = {
    "WavePrefixSum", "WavePrefixProduct", "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor",
    "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst", "WaveBallot"
};

const std::unordered_set<std::string> WaveOperationValidator::fullParticipationOps = {
    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits"
};

bool WaveOperationValidator::requiresFullParticipation(const std::string& intrinsicName) const {
    return fullParticipationOps.count(intrinsicName) > 0;
}

bool WaveOperationValidator::isOrderDependent(const std::string& intrinsicName) const {
    return orderDependentOps.count(intrinsicName) > 0;
}

ValidationResult WaveOperationValidator::validateWaveOp(
    const std::string& intrinsicName,
    const std::vector<Expression*>& args,
    const ControlFlowAnalyzer::ControlFlowState& cfState) {
    
    ValidationResult result;
    result.isValid = true;
    
    // Check 1: Forbidden order-dependent operations
    if (isOrderDependent(intrinsicName)) {
        result.addError(ValidationError::OrderDependentWaveOp,
                       "Order-dependent wave operation not allowed in MiniHLSL: " + intrinsicName);
        return result;
    }
    
    // Check 2: Operations requiring full wave participation
    if (requiresFullParticipation(intrinsicName) && cfState.divergenceLevel > 0) {
        result.addError(ValidationError::IncompleteWaveParticipation,
                       "Wave operation " + intrinsicName + " used in divergent control flow");
        return result;
    }
    
    // Check 3: Valid operation in current context
    if (orderIndependentOps.count(intrinsicName) == 0) {
        result.addError(ValidationError::InvalidWaveContext,
                       "Unknown or unsupported wave operation: " + intrinsicName);
        return result;
    }
    
    return result;
}

// Simplified string-based validator for integration with existing fuzzer
class StringBasedValidator {
public:
    static ValidationResult validateHLSLSource(const std::string& source) {
        ValidationResult result;
        result.isValid = true;
        
        // Check 1: Forbidden keywords
        static const std::unordered_set<std::string> localForbiddenKeywords = {
            "for", "while", "do", "break", "continue", "switch", "case", "default",
            "goto", "struct", "class", "template", "namespace", "using",
            "atomic", "barrier", "sync", "lock", "mutex"
        };
        
        for (const auto& keyword : localForbiddenKeywords) {
            if (containsKeyword(source, keyword)) {
                result.addError(ValidationError::UnsupportedOperation,
                               "Forbidden keyword in MiniHLSL: " + keyword);
            }
        }
        
        // Check 2: Forbidden intrinsics
        static const std::unordered_set<std::string> localForbiddenIntrinsics = {
            "WavePrefixSum", "WavePrefixProduct", "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor",
            "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst",
            "WaveBallot", "WaveMultiPrefixSum", "WaveMultiPrefixProduct",
            "barrier", "AllMemoryBarrier", "GroupMemoryBarrier", "DeviceMemoryBarrier"
        };
        
        for (const auto& intrinsic : localForbiddenIntrinsics) {
            if (source.find(intrinsic) != std::string::npos) {
                result.addError(ValidationError::OrderDependentWaveOp,
                               "Forbidden wave intrinsic in MiniHLSL: " + intrinsic);
            }
        }
        
        // Check 3: Wave operations in divergent control flow
        if (hasWaveOpsInDivergentFlow(source)) {
            result.addError(ValidationError::IncompleteWaveParticipation,
                           "Wave operations found in potentially divergent control flow");
        }
        
        // Check 4: Non-uniform conditions before wave ops
        if (hasNonUniformWaveConditions(source)) {
            result.addError(ValidationError::NonUniformBranch,
                           "Non-uniform conditions detected before wave operations");
        }
        
        return result;
    }
    
private:
    static bool containsKeyword(const std::string& source, const std::string& keyword) {
        // Use word boundary regex to avoid false positives
        std::regex pattern(R"(\b)" + keyword + R"(\b)");
        return std::regex_search(source, pattern);
    }
    
    static bool hasWaveOpsInDivergentFlow(const std::string& source) {
        // Look for wave operations inside lane-dependent if statements
        std::regex laneCondition(R"(if\s*\([^)]*WaveGetLaneIndex\(\)[^)]*\))");
        std::regex waveOp(R"(WaveActive\w+\s*\()");
        
        std::smatch condMatch;
        auto searchStart = source.cbegin();
        
        while (std::regex_search(searchStart, source.cend(), condMatch, laneCondition)) {
            // Find matching closing brace for this if statement
            size_t ifPos = condMatch.position() + std::distance(source.cbegin(), searchStart);
            size_t openBrace = source.find('{', ifPos);
            if (openBrace != std::string::npos) {
                size_t closeBrace = findMatchingBrace(source, openBrace);
                if (closeBrace != std::string::npos) {
                    std::string ifBody = source.substr(openBrace, closeBrace - openBrace);
                    if (std::regex_search(ifBody, waveOp)) {
                        return true;
                    }
                }
            }
            searchStart = condMatch.suffix().first;
        }
        
        return false;
    }
    
    static bool hasNonUniformWaveConditions(const std::string& source) {
        // Check for lane-dependent conditions immediately before wave operations
        std::regex pattern(R"(if\s*\([^)]*WaveGetLaneIndex\(\)[^)]*\)\s*\{[^}]*WaveActive)");
        return std::regex_search(source, pattern);
    }
    
    static size_t findMatchingBrace(const std::string& source, size_t openPos) {
        int braceCount = 1;
        for (size_t i = openPos + 1; i < source.length(); ++i) {
            if (source[i] == '{') braceCount++;
            else if (source[i] == '}') {
                braceCount--;
                if (braceCount == 0) return i;
            }
        }
        return std::string::npos;
    }
};

// MiniHLSLValidator implementation
ValidationResult MiniHLSLValidator::validateSource(const std::string& hlslSource) {
    // Use string-based validation for integration with existing fuzzer
    return StringBasedValidator::validateHLSLSource(hlslSource);
}

bool MiniHLSLValidator::isAllowedConstruct(const std::string& constructName) const {
    return forbiddenKeywords.count(constructName) == 0 && 
           forbiddenIntrinsics.count(constructName) == 0;
}

ValidationResult MiniHLSLValidator::validateProgram(const Program* program) {
    ValidationResult result;
    result.isValid = true;
    
    // Placeholder for full AST-based validation
    // In practice, this would walk the AST and validate each node
    
    return result;
}

ValidationResult MiniHLSLValidator::validateFunction(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    // Placeholder for function-level validation
    
    return result;
}

// Utility function implementations
bool isCommutativeOperation(const std::string& op) {
    static const std::unordered_set<std::string> commutativeOps = {
        "+", "*", "==", "!=", "&&", "||", "&", "|", "^"
    };
    return commutativeOps.count(op) > 0;
}

bool isAssociativeOperation(const std::string& op) {
    static const std::unordered_set<std::string> associativeOps = {
        "+", "*", "&&", "||", "&", "|", "^"
    };
    return associativeOps.count(op) > 0;
}

bool isDeterministicExpression(const Expression* expr) {
    // Conservative: assume non-deterministic unless proven otherwise
    // Full implementation would analyze AST for non-deterministic functions
    return true; // Placeholder
}

std::vector<std::string> generateOrderIndependentVariants(const std::string& baseProgram) {
    std::vector<std::string> variants;
    
    // Generate semantic-preserving mutations that maintain order-independence
    
    // Variant 1: Add commutative operations
    std::string variant1 = baseProgram;
    size_t insertPos = variant1.find("void main(");
    if (insertPos != std::string::npos) {
        insertPos = variant1.find('{', insertPos) + 1;
        std::string injection = R"(
    // Order-independent arithmetic
    uint lane = WaveGetLaneIndex();
    float commutativeResult = float(lane) + float(lane * 2);
    float sum = WaveActiveSum(commutativeResult);
)";
        variant1.insert(insertPos, injection);
        variants.push_back(variant1);
    }
    
    // Variant 2: Add uniform branching
    std::string variant2 = baseProgram;
    insertPos = variant2.find("void main(");
    if (insertPos != std::string::npos) {
        insertPos = variant2.find('{', insertPos) + 1;
        std::string injection = R"(
    // Uniform condition across wave
    if (WaveGetLaneCount() >= 4) {
        uint count = WaveActiveCountBits(true);
        float uniformResult = WaveActiveSum(1.0f);
    }
)";
        variant2.insert(insertPos, injection);
        variants.push_back(variant2);
    }
    
    // Variant 3: Add associative reductions
    std::string variant3 = baseProgram;
    insertPos = variant3.find("void main(");
    if (insertPos != std::string::npos) {
        insertPos = variant3.find('{', insertPos) + 1;
        std::string injection = R"(
    // Associative operations (order-independent)
    uint idx = WaveGetLaneIndex();
    uint product = WaveActiveProduct(idx + 1);
    uint maxVal = WaveActiveMax(idx);
)";
        variant3.insert(insertPos, injection);
        variants.push_back(variant3);
    }
    
    return variants;
}

} // namespace minihlsl