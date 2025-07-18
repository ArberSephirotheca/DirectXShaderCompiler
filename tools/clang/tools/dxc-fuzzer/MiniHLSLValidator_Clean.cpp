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
// NOTE: Removed "for", "while", "switch", "break", "continue" to support these constructs
const std::unordered_set<std::string> minihlsl::MiniHLSLValidator::forbiddenKeywords = {
    "do", "goto", "struct", "class", "template", "namespace", "using",
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

// Control Flow AST Visitor for validating loops and control flow
class ControlFlowASTVisitor : public clang::RecursiveASTVisitor<ControlFlowASTVisitor> {
public:
    explicit ControlFlowASTVisitor(ValidationResult& result) : result_(result), loopDepth_(0), divergenceLevel_(0) {}
    
    // Visit for loops
    bool VisitForStmt(clang::ForStmt* forStmt) {
        loopDepth_++;
        
        // Check if loop condition is uniform
        if (forStmt->getCond() && !isUniformCondition(forStmt->getCond())) {
            divergenceLevel_++;
            
            // Check for wave operations in divergent loop
            if (hasWaveOperationsInStmt(forStmt->getBody())) {
                result_.addError(ValidationError::IncompleteWaveParticipation,
                               "Wave operations found in divergent for loop");
            }
            
            divergenceLevel_--;
        }
        
        // Continue traversing the loop body
        TraverseStmt(forStmt->getBody());
        
        loopDepth_--;
        return true;
    }
    
    // Visit while loops
    bool VisitWhileStmt(clang::WhileStmt* whileStmt) {
        loopDepth_++;
        
        // Check if loop condition is uniform
        if (whileStmt->getCond() && !isUniformCondition(whileStmt->getCond())) {
            divergenceLevel_++;
            
            // Check for wave operations in divergent loop
            if (hasWaveOperationsInStmt(whileStmt->getBody())) {
                result_.addError(ValidationError::IncompleteWaveParticipation,
                               "Wave operations found in divergent while loop");
            }
            
            divergenceLevel_--;
        }
        
        // Continue traversing the loop body
        TraverseStmt(whileStmt->getBody());
        
        loopDepth_--;
        return true;
    }
    
    // Visit switch statements
    bool VisitSwitchStmt(clang::SwitchStmt* switchStmt) {
        // Check if switch condition is uniform
        if (switchStmt->getCond() && !isUniformCondition(switchStmt->getCond())) {
            divergenceLevel_++;
            
            // Check for wave operations in divergent switch
            if (hasWaveOperationsInStmt(switchStmt->getBody())) {
                result_.addError(ValidationError::IncompleteWaveParticipation,
                               "Wave operations found in divergent switch statement");
            }
            
            divergenceLevel_--;
        }
        
        return true;
    }
    
    // Visit break statements
    bool VisitBreakStmt(clang::BreakStmt* breakStmt) {
        if (loopDepth_ == 0) {
            result_.addError(ValidationError::InvalidExpression,
                           "Break statement outside of loop or switch");
        }
        return true;
    }
    
    // Visit continue statements
    bool VisitContinueStmt(clang::ContinueStmt* continueStmt) {
        if (loopDepth_ == 0) {
            result_.addError(ValidationError::InvalidExpression,
                           "Continue statement outside of loop");
        }
        return true;
    }
    
    // Visit if statements (existing logic)
    bool VisitIfStmt(clang::IfStmt* ifStmt) {
        if (ifStmt->getCond() && !isUniformCondition(ifStmt->getCond())) {
            divergenceLevel_++;
            
            if (hasWaveOperationsInStmt(ifStmt->getThen()) || 
                (ifStmt->getElse() && hasWaveOperationsInStmt(ifStmt->getElse()))) {
                result_.addError(ValidationError::IncompleteWaveParticipation,
                               "Wave operations found in divergent if statement");
            }
            
            divergenceLevel_--;
        }
        
        return true;
    }
    
    // Visit function calls to detect wave operations
    bool VisitCallExpr(clang::CallExpr* callExpr) {
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
    int loopDepth_;
    int divergenceLevel_;
    
    bool isUniformCondition(clang::Expr* expr) {
        // Check if expression is uniform across all lanes
        // This is a simplified check - in production, you'd track variable uniformity
        return isDeterministicClangExpr(expr);
    }
    
    bool hasWaveOperationsInStmt(clang::Stmt* stmt) {
        if (!stmt) return false;
        
        // Check if this statement is a wave operation call
        if (auto* callExpr = clang::dyn_cast<clang::CallExpr>(stmt)) {
            if (const auto* declRef = clang::dyn_cast<clang::DeclRefExpr>(callExpr->getCallee())) {
                if (const auto* funcDecl = clang::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
                    std::string funcName = funcDecl->getNameAsString();
                    return funcName.find("Wave") == 0; // Simple heuristic
                }
            }
        }
        
        // Recursively check children
        for (auto* child : stmt->children()) {
            if (hasWaveOperationsInStmt(child)) {
                return true;
            }
        }
        
        return false;
    }
};

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
        return result;
    }
    
    // Get function body
    const clang::Stmt* body = func->getBody();
    if (!body) {
        // Function has no body (declaration only)
        return result;
    }
    
    // Use our enhanced AST visitor to validate control flow
    ControlFlowASTVisitor visitor(result);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(body));
    
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
    
    // Variant 1: Basic order-independent operations
    std::string variant1 = baseProgram;
    size_t pos = variant1.find("void main(");
    if (pos != std::string::npos) {
        pos = variant1.find('{', pos) + 1;
        std::string injection = "\n    // Order-independent operations\n";
        injection += "    uint lane = WaveGetLaneIndex();\n";
        injection += "    float sum = WaveActiveSum(float(lane));\n";
        variant1.insert(pos, injection);
        variants.push_back(variant1);
    }
    
    // Variant 2: For loop with uniform condition
    std::string variant2 = baseProgram;
    pos = variant2.find("void main(");
    if (pos != std::string::npos) {
        pos = variant2.find('{', pos) + 1;
        std::string injection = "\n    // For loop with uniform condition\n";
        injection += "    float total = 0.0f;\n";
        injection += "    for (uint i = 0; i < 4; i++) {\n";
        injection += "        total += float(i);\n";
        injection += "    }\n";
        injection += "    float waveTotal = WaveActiveSum(total);\n";
        variant2.insert(pos, injection);
        variants.push_back(variant2);
    }
    
    // Variant 3: While loop with uniform condition
    std::string variant3 = baseProgram;
    pos = variant3.find("void main(");
    if (pos != std::string::npos) {
        pos = variant3.find('{', pos) + 1;
        std::string injection = "\n    // While loop with uniform condition\n";
        injection += "    uint counter = 0;\n";
        injection += "    while (counter < 3) {\n";
        injection += "        counter++;\n";
        injection += "        if (counter == 2) break;\n";
        injection += "    }\n";
        injection += "    uint waveCount = WaveActiveSum(counter);\n";
        variant3.insert(pos, injection);
        variants.push_back(variant3);
    }
    
    // Variant 4: Switch statement with uniform condition
    std::string variant4 = baseProgram;
    pos = variant4.find("void main(");
    if (pos != std::string::npos) {
        pos = variant4.find('{', pos) + 1;
        std::string injection = "\n    // Switch with uniform condition\n";
        injection += "    uint mode = 1;\n";
        injection += "    float value = 0.0f;\n";
        injection += "    switch (mode) {\n";
        injection += "        case 0:\n";
        injection += "            value = 1.0f;\n";
        injection += "            break;\n";
        injection += "        case 1:\n";
        injection += "            value = 2.0f;\n";
        injection += "            break;\n";
        injection += "        default:\n";
        injection += "            value = 0.0f;\n";
        injection += "            break;\n";
        injection += "    }\n";
        injection += "    float waveValue = WaveActiveSum(value);\n";
        variant4.insert(pos, injection);
        variants.push_back(variant4);
    }
    
    // Variant 5: Nested control flow
    std::string variant5 = baseProgram;
    pos = variant5.find("void main(");
    if (pos != std::string::npos) {
        pos = variant5.find('{', pos) + 1;
        std::string injection = "\n    // Nested control flow\n";
        injection += "    float result = 0.0f;\n";
        injection += "    for (uint i = 0; i < 2; i++) {\n";
        injection += "        if (i == 0) {\n";
        injection += "            result += 1.0f;\n";
        injection += "        } else {\n";
        injection += "            result += 2.0f;\n";
        injection += "        }\n";
        injection += "    }\n";
        injection += "    float waveResult = WaveActiveSum(result);\n";
        variant5.insert(pos, injection);
        variants.push_back(variant5);
    }
    
    return variants;
}

} // namespace minihlsl