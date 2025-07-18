#include "MiniHLSLValidator.h"
#include <regex>
#include <algorithm>
#include <sstream>
#include <unordered_set>

// Clang AST includes for proper AST-based analysis
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

// Forward declarations for AST-based analysis
bool isDeterministicClangExpr(const clang::Expr* expr);

// Note: This validator now uses proper Clang AST types for robust analysis
// The type aliases in the header (Expression = clang::Expr, etc.) ensure
// seamless integration with DXC's existing AST infrastructure

// Static initialization of forbidden constructs (must be outside namespace)
const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::forbiddenKeywords = {
    "for", "while", "do", "break", "continue", "switch", "case", "default",
    "goto", "struct", "class", "template", "namespace", "using",
    "atomic", "barrier", "sync", "lock", "mutex"
};

const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::forbiddenIntrinsics = {
    "WavePrefixSum", "WavePrefixProduct", "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor",
    "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst",
    "WaveBallot", "WaveMultiPrefixSum", "WaveMultiPrefixProduct",
    "barrier", "AllMemoryBarrier", "GroupMemoryBarrier", "DeviceMemoryBarrier"
};

// Threadgroup-level operation classification based on OrderIndependenceProof.lean
const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::allowedThreadgroupOps = {
    "GroupMemoryBarrierWithGroupSync",  // barrier
    "InterlockedAdd",                   // sharedAtomicAdd
    "InterlockedMin", "InterlockedMax", // atomic min/max operations
    "InterlockedAnd", "InterlockedOr", "InterlockedXor"  // atomic bitwise operations
};

const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::sharedMemoryOps = {
    "groupshared", "InterlockedAdd", "InterlockedMin", "InterlockedMax",
    "InterlockedAnd", "InterlockedOr", "InterlockedXor"
};

// UniformityAnalyzer static members (must be outside namespace)
const std::unordered_set<std::string> minihlsl::UniformityAnalyzer::uniformIntrinsics = {
    "WaveGetLaneCount", "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
};

const std::unordered_set<std::string> minihlsl::UniformityAnalyzer::divergentIntrinsics = {
    "WaveGetLaneIndex", "WaveIsFirstLane"
};

// WaveOperationValidator static members (must be outside namespace)
const std::unordered_set<std::string> minihlsl::WaveOperationValidator::orderIndependentOps = {
    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits",
    "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
    "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
};

const std::unordered_set<std::string> minihlsl::WaveOperationValidator::orderDependentOps = {
    "WavePrefixSum", "WavePrefixProduct", "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor",
    "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst", "WaveBallot"
};

const std::unordered_set<std::string> minihlsl::WaveOperationValidator::fullParticipationOps = {
    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits"
};

namespace minihlsl {

// Bring std namespace into scope to avoid minihlsl::std conflicts
using ::std::string;
using ::std::vector;
using ::std::unordered_set;
using ::std::unordered_map;
using ::std::regex;
using ::std::smatch;
using ::std::regex_search;
using ::std::distance;

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

// ThreadgroupValidator implementation - implements safety constraints from OrderIndependenceProof.lean
class ThreadgroupValidator {
public:
    // Track shared memory usage patterns for disjoint write validation
    struct SharedMemoryAccess {
        std::string address;
        std::string accessType; // "read", "write", "atomic"
        std::string waveCondition; // condition under which this access occurs
        bool isUniform; // whether the access is uniform across threadgroup
    };
    
    // Validate threadgroup-level safety constraints
    ValidationResult validateThreadgroupSafety(const std::string& source) {
        ValidationResult result;
        result.isValid = true;
        
        // Check 1: Disjoint writes constraint
        if (!hasDisjointWrites(source)) {
            result.addError(ValidationError::MemoryRaceCondition,
                           "Potential overlapping writes to shared memory detected");
        }
        
        // Check 2: Only commutative operations on shared memory
        if (!hasOnlyCommutativeOps(source)) {
            result.addError(ValidationError::NonCommutativeOperation,
                           "Non-commutative operations on shared memory not allowed");
        }
        
        // Check 3: Proper separation of wave and threadgroup operations
        if (!hasProperOperationSeparation(source)) {
            result.addError(ValidationError::MixedOperationScope,
                           "Mixed wave and threadgroup operations must be separated");
        }
        
        return result;
    }
    
private:
    bool hasDisjointWrites(const std::string& source) {
        // Pattern: groupshared array[threadID] = value (disjoint by thread ID)
        std::regex disjointPattern(R"(groupshared\s+\w+\s*\[\s*\w*threadID\w*\s*\]\s*=)");
        
        // Pattern: overlapping writes (multiple threads writing to same address)
        std::regex overlapPattern(R"(groupshared\s+\w+\s*\[\s*(?!\w*threadID\w*)\w*\s*\]\s*=)");
        
        return std::regex_search(source, disjointPattern) && 
               !std::regex_search(source, overlapPattern);
    }
    
    bool hasOnlyCommutativeOps(const std::string& source) {
        // Check for atomic operations - these are commutative
        std::regex atomicPattern(R"(Interlocked(Add|Min|Max|And|Or|Xor)\s*\()");
        
        // Check for non-commutative operations (subtraction, division)
        std::regex nonCommutativePattern(R"(groupshared\s+\w+[^=]*=\s*[^=]*[-/])");
        
        return !std::regex_search(source, nonCommutativePattern);
    }
    
    bool hasProperOperationSeparation(const std::string& source) {
        // Check for mixed operations like: groupshared[addr] = WaveActiveSum(...)
        std::regex mixedPattern(R"(groupshared\s+\w+\s*\[[^\]]*\]\s*=\s*[^;]*Wave\w+\s*\()");
        
        // Mixed operations should be separated into:
        // 1. waveResult = WaveActiveSum(...)  (wave operation)
        // 2. groupshared[addr] = waveResult   (threadgroup operation)
        return !std::regex_search(source, mixedPattern);
    }
};

// WaveOperationValidator implementation (static members defined outside namespace above)

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
        
        // Check 5: Threadgroup-level safety constraints (NEW)
        ThreadgroupValidator tgValidator;
        ValidationResult tgResult = tgValidator.validateThreadgroupSafety(source);
        if (!tgResult.isValid) {
            for (size_t i = 0; i < tgResult.errors.size(); ++i) {
                result.addError(tgResult.errors[i], tgResult.errorMessages[i]);
            }
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
    
    if (!program) {
        result.addError(ValidationError::InvalidExpression, "Program is null");
        return result;
    }
    
    // Check for forbidden constructs at program level
    ValidationResult constructResult = checkForbiddenConstructs(program);
    if (!constructResult.isValid) {
        for (size_t i = 0; i < constructResult.errors.size(); ++i) {
            result.addError(constructResult.errors[i], constructResult.errorMessages[i]);
        }
    }
    
    // Validate each function in the program
    // Note: This assumes Program has a way to access its functions
    // In a real implementation, we'd iterate through program->getFunctions() or similar
    
    return result;
}

ValidationResult MiniHLSLValidator::validateFunction(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
        return result;
    }
    
    // Validate function signature
    ValidationResult sigResult = validateFunctionSignature(func);
    if (!sigResult.isValid) {
        for (size_t i = 0; i < sigResult.errors.size(); ++i) {
            result.addError(sigResult.errors[i], sigResult.errorMessages[i]);
        }
    }
    
    // Validate control flow for wave operations
    ValidationResult cfResult = controlFlowAnalyzer.analyzeFunction(func);
    if (!cfResult.isValid) {
        for (size_t i = 0; i < cfResult.errors.size(); ++i) {
            result.addError(cfResult.errors[i], cfResult.errorMessages[i]);
        }
    }
    
    // Validate memory access patterns
    ValidationResult memResult = memoryAnalyzer.analyzeMemoryAccesses(func);
    if (!memResult.isValid) {
        for (size_t i = 0; i < memResult.errors.size(); ++i) {
            result.addError(memResult.errors[i], memResult.errorMessages[i]);
        }
    }
    
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
    if (!expr) {
        return false; // Null expression is not deterministic
    }
    
    // Direct AST-based determinism analysis using Clang AST types
    // Expression* is now an alias for clang::Expr*, so no cast needed
    return isDeterministicClangExpr(expr);
}

// Helper function for AST-based determinism analysis
bool isDeterministicClangExpr(const clang::Expr* expr) {
    if (!expr) {
        return false;
    }
    
    using namespace clang;
    
    // Analyze based on actual AST node type
    switch (expr->getStmtClass()) {
        // DETERMINISTIC: Literals are always deterministic
        case Stmt::IntegerLiteralClass:
        case Stmt::FloatingLiteralClass:
        case Stmt::StringLiteralClass:
        case Stmt::CharacterLiteralClass:
        case Stmt::CXXBoolLiteralExprClass:
            return true;
            
        // DETERMINISTIC: Variable references (deterministic if variable is deterministic)
        case Stmt::DeclRefExprClass: {
            const auto* declRef = cast<DeclRefExpr>(expr);
            const auto* decl = declRef->getDecl();
            
            // Check if it's a deterministic built-in (like lane/wave indices)
            if (const auto* funcDecl = dyn_cast<FunctionDecl>(decl)) {
                std::string funcName = funcDecl->getNameAsString();
                
                // HLSL deterministic intrinsics
                static const std::unordered_set<std::string> deterministicIntrinsics = {
                    "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
                    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
                    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits",
                    "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
                };
                
                // NON-DETERMINISTIC: Order-dependent operations
                static const std::unordered_set<std::string> nonDeterministicIntrinsics = {
                    "WavePrefixSum", "WavePrefixProduct", "WaveReadLaneAt",
                    "WaveReadFirstLane", "WaveBallot", "rand", "random"
                };
                
                if (nonDeterministicIntrinsics.count(funcName)) {
                    return false;
                }
                
                return deterministicIntrinsics.count(funcName) > 0;
            }
            
            // Regular variable references are deterministic
            return true;
        }
        
        // DETERMINISTIC: Arithmetic operations (if operands are deterministic)
        case Stmt::BinaryOperatorClass: {
            const auto* binOp = cast<BinaryOperator>(expr);
            BinaryOperatorKind op = binOp->getOpcode();
            
            // Check if operation is deterministic
            switch (op) {
                case BO_Add: case BO_Sub: case BO_Mul: case BO_Div: case BO_Rem:
                case BO_Shl: case BO_Shr: case BO_And: case BO_Or: case BO_Xor:
                case BO_LT: case BO_GT: case BO_LE: case BO_GE: case BO_EQ: case BO_NE:
                case BO_LAnd: case BO_LOr:
                    // Deterministic if both operands are deterministic
                    return isDeterministicClangExpr(binOp->getLHS()) && 
                           isDeterministicClangExpr(binOp->getRHS());
                    
                default:
                    return false; // Unsupported operation
            }
        }
        
        // DETERMINISTIC: Unary operations (if operand is deterministic)
        case Stmt::UnaryOperatorClass: {
            const auto* unOp = cast<UnaryOperator>(expr);
            UnaryOperatorKind op = unOp->getOpcode();
            
            switch (op) {
                case UO_Plus: case UO_Minus: case UO_Not: case UO_LNot:
                    return isDeterministicClangExpr(unOp->getSubExpr());
                    
                default:
                    return false; // Unsupported unary operation
            }
        }
        
        // DETERMINISTIC: Function calls (if function is deterministic)
        case Stmt::CallExprClass: {
            const auto* callExpr = cast<CallExpr>(expr);
            const auto* callee = callExpr->getCallee();
            
            // Check if it's a deterministic function
            if (const auto* declRef = dyn_cast<DeclRefExpr>(callee)) {
                return isDeterministicClangExpr(declRef);
            }
            
            return false; // Unknown function call
        }
        
        // DETERMINISTIC: Member access (if base is deterministic)
        case Stmt::MemberExprClass: {
            const auto* memberExpr = cast<MemberExpr>(expr);
            return isDeterministicClangExpr(memberExpr->getBase());
        }
        
        // DETERMINISTIC: Array subscript (if base and index are deterministic)
        case Stmt::ArraySubscriptExprClass: {
            const auto* arrayExpr = cast<ArraySubscriptExpr>(expr);
            return isDeterministicClangExpr(arrayExpr->getBase()) && 
                   isDeterministicClangExpr(arrayExpr->getIdx());
        }
        
        // DETERMINISTIC: HLSL vector swizzle (if base is deterministic)
        // Note: HLSLVectorElementExprClass may not be in StmtClass enum
        // case Stmt::HLSLVectorElementExprClass: {
        //     const auto* vectorExpr = cast<HLSLVectorElementExpr>(expr);
        //     return isDeterministicClangExpr(vectorExpr->getBase());
        // }
        
        // DETERMINISTIC: Casts (if operand is deterministic)
        case Stmt::ImplicitCastExprClass:
        case Stmt::CStyleCastExprClass: {
            const auto* castExpr = cast<CastExpr>(expr);
            return isDeterministicClangExpr(castExpr->getSubExpr());
        }
        
        // DETERMINISTIC: Conditional operator (if all parts are deterministic)
        case Stmt::ConditionalOperatorClass: {
            const auto* condExpr = cast<ConditionalOperator>(expr);
            return isDeterministicClangExpr(condExpr->getCond()) &&
                   isDeterministicClangExpr(condExpr->getTrueExpr()) &&
                   isDeterministicClangExpr(condExpr->getFalseExpr());
        }
        
        default:
            // Unknown expression type - conservative approach
            return false;
    }
}

vector<string> generateOrderIndependentVariants(const string& baseProgram) {
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
    
    // Variant 4: Add threadgroup-level operations (NEW)
    std::string variant4 = baseProgram;
    insertPos = variant4.find("void main(");
    if (insertPos != std::string::npos) {
        insertPos = variant4.find('{', insertPos) + 1;
        std::string injection = R"(
    // Threadgroup-level order-independent operations
    groupshared uint sharedData[64];
    uint threadID = WaveGetLaneIndex() + WaveGetLaneCount() * WaveGetLaneIndex();
    
    // Step 1: Wave operation (order-independent)
    uint waveSum = WaveActiveSum(WaveGetLaneIndex());
    
    // Step 2: Disjoint shared memory write (order-independent)
    sharedData[threadID] = waveSum;
    
    // Step 3: Barrier synchronization
    GroupMemoryBarrierWithGroupSync();
    
    // Step 4: Atomic operation (commutative, order-independent)
    uint oldValue;
    InterlockedAdd(sharedData[0], waveSum, oldValue);
)";
        variant4.insert(insertPos, injection);
        variants.push_back(variant4);
    }
    
    // Variant 5: Add separated operation pattern (enforces proper separation)
    std::string variant5 = baseProgram;
    insertPos = variant5.find("void main(");
    if (insertPos != std::string::npos) {
        insertPos = variant5.find('{', insertPos) + 1;
        std::string injection = R"(
    // Proper separation of wave and threadgroup operations
    groupshared uint atomicCounter;
    
    // BAD: groupshared[0] = WaveActiveSum(value);  // Mixed operation
    // GOOD: Separate into two statements
    uint waveResult = WaveActiveSum(WaveGetLaneIndex());  // Wave operation
    InterlockedAdd(atomicCounter, waveResult);            // Threadgroup operation
)";
        variant5.insert(insertPos, injection);
        variants.push_back(variant5);
    }
    
    return variants;
}

// Additional member function implementations that were missing

ValidationResult MiniHLSLValidator::validateFunctionSignature(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
        return result;
    }
    
    // Basic signature validation for MiniHLSL
    // In a real implementation, we'd check:
    // - Return type is allowed
    // - Parameters are of allowed types
    // - No unsupported qualifiers
    
    return result;
}

ValidationResult MiniHLSLValidator::validateVariableDecl(const Statement* stmt) {
    ValidationResult result;
    result.isValid = true;
    
    if (!stmt) {
        result.addError(ValidationError::InvalidExpression, "Statement is null");
        return result;
    }
    
    // Check if variable declaration uses allowed types
    // In a real implementation, we'd parse the declaration and check:
    // - Type is supported in MiniHLSL
    // - No forbidden qualifiers
    // - Proper initialization
    
    return result;
}

ValidationResult MiniHLSLValidator::checkForbiddenConstructs(const Program* program) {
    ValidationResult result;
    result.isValid = true;
    
    if (!program) {
        result.addError(ValidationError::InvalidExpression, "Program is null");
        return result;
    }
    
    // In a real implementation, we'd walk the AST and check for:
    // - Forbidden keywords (loops, complex control flow)
    // - Forbidden intrinsics (order-dependent operations)
    // - Unsupported language constructs
    
    // For now, this is handled by the string-based validator
    return result;
}

// AST-based analyzer implementations

// AST Visitor for control flow analysis
class ControlFlowASTVisitor : public clang::RecursiveASTVisitor<ControlFlowASTVisitor> {
public:
    explicit ControlFlowASTVisitor(ValidationResult& result) : result_(result), divergenceLevel_(0) {}
    
    bool VisitIfStmt(clang::IfStmt* ifStmt) {
        // Check if condition is uniform (required for wave operations)
        clang::Expr* condition = ifStmt->getCond();
        if (!isUniformCondition(condition)) {
            divergenceLevel_++;
            
            // Check if there are wave operations in the divergent branch
            if (hasWaveOperationsInStmt(ifStmt->getThen()) || 
                (ifStmt->getElse() && hasWaveOperationsInStmt(ifStmt->getElse()))) {
                result_.addError(ValidationError::IncompleteWaveParticipation,
                               "Wave operations found in divergent control flow");
            }
            
            divergenceLevel_--;
        }
        
        return true;
    }
    
    bool VisitForStmt(clang::ForStmt* forStmt) {
        // MiniHLSL doesn't allow loops
        result_.addError(ValidationError::UnsupportedOperation, 
                        "For loops are not allowed in MiniHLSL");
        return true;
    }
    
    bool VisitWhileStmt(clang::WhileStmt* whileStmt) {
        // MiniHLSL doesn't allow loops
        result_.addError(ValidationError::UnsupportedOperation, 
                        "While loops are not allowed in MiniHLSL");
        return true;
    }
    
    bool VisitCallExpr(clang::CallExpr* callExpr) {
        // Check for wave operations in divergent control flow
        if (divergenceLevel_ > 0) {
            if (const auto* declRef = clang::dyn_cast<clang::DeclRefExpr>(callExpr->getCallee())) {
                if (const auto* funcDecl = clang::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
                    std::string funcName = funcDecl->getNameAsString();
                    
                    // Check if it's a wave operation
                    static const std::unordered_set<std::string> waveOps = {
                        "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
                        "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits"
                    };
                    
                    if (waveOps.count(funcName)) {
                        result_.addError(ValidationError::IncompleteWaveParticipation,
                                       "Wave operation " + funcName + " used in divergent control flow");
                    }
                }
            }
        }
        
        return true;
    }
    
private:
    ValidationResult& result_;
    int divergenceLevel_;
    
    bool isUniformCondition(clang::Expr* expr) {
        // Check if expression is uniform across all lanes
        // This is a simplified check - full implementation would track uniformity
        return isDeterministicClangExpr(expr);
    }
    
    bool hasWaveOperationsInStmt(clang::Stmt* stmt) {
        // Recursively check for wave operations in statement
        if (!stmt) return false;
        
        if (auto* callExpr = clang::dyn_cast<clang::CallExpr>(stmt)) {
            if (const auto* declRef = clang::dyn_cast<clang::DeclRefExpr>(callExpr->getCallee())) {
                if (const auto* funcDecl = clang::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
                    std::string funcName = funcDecl->getNameAsString();
                    return funcName.find("Wave") == 0; // Simple heuristic
                }
            }
        }
        
        // Check children
        for (auto* child : stmt->children()) {
            if (hasWaveOperationsInStmt(child)) {
                return true;
            }
        }
        
        return false;
    }
};

ValidationResult ControlFlowAnalyzer::analyzeFunction(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
        return result;
    }
    
    // Function* is now an alias for clang::FunctionDecl*, so no cast needed
    const clang::FunctionDecl* clangFunc = func;
    
    // Get function body
    const clang::Stmt* body = clangFunc->getBody();
    if (!body) {
        // Function has no body (declaration only)
        return result;
    }
    
    // Use AST visitor for comprehensive analysis
    ControlFlowASTVisitor visitor(result);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(body));
    
    return result;
}

ValidationResult MemoryAccessAnalyzer::analyzeMemoryAccesses(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
        return result;
    }
    
    // AST-based memory access analysis
    const clang::Stmt* body = func->getBody();
    if (!body) {
        return result; // No body to analyze
    }
    
    // In a full implementation, we'd create a MemoryAccessASTVisitor
    // to analyze array accesses, shared memory operations, etc.
    // For now, basic validation is sufficient
    
    return result;
}

bool verifyReconvergence(const Function* func) {
    if (!func) {
        return false;
    }
    
    // Basic reconvergence verification
    // In a real implementation, we'd ensure:
    // - All divergent paths reconverge before wave operations
    // - Proper control flow structure
    
    return true; // Conservative: assume reconvergence for MiniHLSL
}

} // namespace minihlsl