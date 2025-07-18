#include "MiniHLSLValidator.h"
#include <regex>
#include <algorithm>
#include <sstream>

// Clang AST includes for proper AST-based analysis
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

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

const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::allowedThreadgroupOps = {
    "GroupMemoryBarrierWithGroupSync", "InterlockedAdd", "InterlockedMin", "InterlockedMax",
    "InterlockedAnd", "InterlockedOr", "InterlockedXor"
};

const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::sharedMemoryOps = {
    "groupshared", "InterlockedAdd", "InterlockedMin", "InterlockedMax",
    "InterlockedAnd", "InterlockedOr", "InterlockedXor"
};

const std::unordered_set<std::string> minihlsl::UniformityAnalyzer::uniformIntrinsics = {
    "WaveGetLaneCount", "WaveActiveAllEqual", "WaveActiveAllTrue", "WaveActiveAnyTrue"
};

const std::unordered_set<std::string> minihlsl::UniformityAnalyzer::divergentIntrinsics = {
    "WaveGetLaneIndex", "WaveIsFirstLane"
};

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

// Forward declaration for AST-based analysis
bool isDeterministicClangExpr(const clang::Expr* expr);

// AST-based expression determinism analysis
bool isDeterministicClangExpr(const clang::Expr* expr) {
    if (!expr) return false;
    
    using namespace clang;
    
    switch (expr->getStmtClass()) {
        case Stmt::IntegerLiteralClass:
        case Stmt::FloatingLiteralClass:
        case Stmt::StringLiteralClass:
        case Stmt::CharacterLiteralClass:
        case Stmt::CXXBoolLiteralExprClass:
            return true;
            
        case Stmt::DeclRefExprClass: {
            const auto* declRef = cast<DeclRefExpr>(expr);
            const auto* decl = declRef->getDecl();
            
            if (const auto* funcDecl = dyn_cast<FunctionDecl>(decl)) {
                std::string funcName = funcDecl->getNameAsString();
                
                static const std::unordered_set<std::string> deterministicIntrinsics = {
                    "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
                    "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
                    "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor", "WaveActiveCountBits"
                };
                
                static const std::unordered_set<std::string> nonDeterministicIntrinsics = {
                    "WavePrefixSum", "WavePrefixProduct", "WaveReadLaneAt", "WaveBallot"
                };
                
                if (nonDeterministicIntrinsics.count(funcName)) return false;
                return deterministicIntrinsics.count(funcName) > 0;
            }
            return true;
        }
        
        case Stmt::BinaryOperatorClass: {
            const auto* binOp = cast<BinaryOperator>(expr);
            return isDeterministicClangExpr(binOp->getLHS()) && 
                   isDeterministicClangExpr(binOp->getRHS());
        }
        
        case Stmt::UnaryOperatorClass: {
            const auto* unOp = cast<UnaryOperator>(expr);
            return isDeterministicClangExpr(unOp->getSubExpr());
        }
        
        default:
            return false;
    }
}

// UniformityAnalyzer implementation
bool UniformityAnalyzer::isUniform(const Expression* expr) const {
    return false; // Conservative approach
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
    
    if (isOrderDependent(intrinsicName)) {
        result.addError(ValidationError::OrderDependentWaveOp,
                       "Order-dependent wave operation: " + intrinsicName);
        return result;
    }
    
    if (requiresFullParticipation(intrinsicName) && cfState.divergenceLevel > 0) {
        result.addError(ValidationError::IncompleteWaveParticipation,
                       "Wave operation in divergent flow: " + intrinsicName);
        return result;
    }
    
    return result;
}

// MiniHLSLValidator implementation
ValidationResult MiniHLSLValidator::validateSource(const std::string& hlslSource) {
    ValidationResult result;
    result.isValid = true;
    
    // Check forbidden keywords
    for (const auto& keyword : forbiddenKeywords) {
        std::regex pattern(R"(\b)" + keyword + R"(\b)");
        if (std::regex_search(hlslSource, pattern)) {
            result.addError(ValidationError::UnsupportedOperation,
                           "Forbidden keyword: " + keyword);
        }
    }
    
    // Check forbidden intrinsics
    for (const auto& intrinsic : forbiddenIntrinsics) {
        if (hlslSource.find(intrinsic) != std::string::npos) {
            result.addError(ValidationError::OrderDependentWaveOp,
                           "Forbidden intrinsic: " + intrinsic);
        }
    }
    
    return result;
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
    }
    
    return result;
}

ValidationResult MiniHLSLValidator::validateFunction(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
    }
    
    return result;
}

// Control flow analyzer implementation
ValidationResult ControlFlowAnalyzer::analyzeFunction(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
    }
    
    return result;
}

// Memory access analyzer implementation
ValidationResult MemoryAccessAnalyzer::analyzeMemoryAccesses(const Function* func) {
    ValidationResult result;
    result.isValid = true;
    
    if (!func) {
        result.addError(ValidationError::InvalidExpression, "Function is null");
    }
    
    return result;
}

// Utility functions
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
    if (!expr) return false;
    return isDeterministicClangExpr(expr);
}

bool verifyReconvergence(const Function* func) {
    return func != nullptr; // Basic implementation
}

std::vector<std::string> generateOrderIndependentVariants(const std::string& baseProgram) {
    std::vector<std::string> variants;
    
    // Basic variant generation
    std::string variant = baseProgram;
    size_t pos = variant.find("void main(");
    if (pos != std::string::npos) {
        pos = variant.find('{', pos) + 1;
        std::string injection = "\n    // Order-independent operations\n";
        injection += "    uint lane = WaveGetLaneIndex();\n";
        injection += "    float sum = WaveActiveSum(float(lane));\n";
        variant.insert(pos, injection);
        variants.push_back(variant);
    }
    
    return variants;
}

} // namespace minihlsl