#pragma once

// Core validation types and enums
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <string>

namespace minihlsl {

// Type aliases for cleaner API
using Expression = clang::Expr;
using Statement = clang::Stmt;
using Function = clang::FunctionDecl;
using Program = clang::TranslationUnitDecl;

// Validation result types
enum class ValidationError {
    // Deterministic control flow violations
    NonDeterministicCondition,         // Control flow condition not compile-time deterministic
    InvalidDeterministicExpression,    // Expression not compile-time deterministic
    MixedDeterministicContext,         // Mixing deterministic and non-deterministic constructs
    
    // Wave operation violations  
    OrderDependentWaveOp,              // Use of order-dependent operations (forbidden)
    IncompleteWaveParticipation,       // Wave operation in divergent context
    InvalidWaveContext,                // Wave operation in invalid control flow context
    
    // Memory access violations
    OverlappingMemoryWrites,           // Violation of hasDisjointWrites constraint
    NonCommutativeMemoryOp,            // Violation of hasOnlyCommutativeOps constraint
    SharedMemoryRaceCondition,         // Race condition in shared memory access
    
    // Type/syntax violations
    UnsupportedType,
    UnsupportedOperation,
    InvalidExpression,
    
    // General violations
    NonDeterministicOperation,
    SideEffectInPureFunction,
    ForbiddenLanguageConstruct
};

struct ValidationResult {
    bool isValid;
    std::vector<ValidationError> errors;
    std::vector<std::string> errorMessages;
    
    void addError(ValidationError error, const std::string& message) {
        isValid = false;
        errors.push_back(error);
        errorMessages.push_back(message);
    }
    
    ValidationResult() : isValid(true) {}
};

// Formal verification integration utilities
struct FormalProofMapping {
    bool deterministicControlFlow;    // Maps to hasDetministicControlFlow
    bool orderIndependentOperations;  // Maps to wave/threadgroup operation constraints
    bool memorySafetyConstraints;     // Maps to hasDisjointWrites & hasOnlyCommutativeOps
    bool programOrderIndependence;    // Overall conclusion
};

// Forward declarations
class DeterministicExpressionAnalyzer;
class ControlFlowAnalyzer;
class MemorySafetyAnalyzer;
class WaveOperationValidator;

// Production-ready implementation of all AST-based validation components
// This provides complete implementations of all validation methods

// Standalone deterministic expression analyzer
class DeterministicExpressionAnalyzer {
public:
    enum class ExpressionKind {
        Literal,                    // Constants: 42, 3.14f, true
        LaneIndex,                  // WaveGetLaneIndex()
        WaveIndex,                  // Current wave ID
        ThreadIndex,                // Global thread index
        WaveProperty,               // WaveGetLaneCount(), etc.
        Arithmetic,                 // +, -, *, / of deterministic expressions
        Comparison,                 // <, >, ==, != of deterministic expressions
        NonDeterministic            // Everything else
    };

    explicit DeterministicExpressionAnalyzer(clang::ASTContext& context) 
        : context_(context) {}
    
    // Core implementations
    bool isCompileTimeDeterministic(const clang::Expr* expr);
    ExpressionKind classifyExpression(const clang::Expr* expr);
    ValidationResult validateDeterministicExpression(const clang::Expr* expr);
    
    // Complete helper method implementations
    bool isLiteralConstant(const clang::Expr* expr);
    bool isLaneIndexExpression(const clang::Expr* expr);
    bool isWavePropertyExpression(const clang::Expr* expr);
    bool isThreadIndexExpression(const clang::Expr* expr);
    bool isArithmeticOfDeterministic(const clang::Expr* expr);
    bool isComparisonOfDeterministic(const clang::Expr* expr);
    
private:
    // Advanced expression analysis methods
    bool analyzeComplexExpression(const clang::Expr* expr);
    bool analyzeConditionalOperator(const clang::ConditionalOperator* cond);
    bool analyzeCastExpression(const clang::CastExpr* cast);
    bool analyzeArraySubscript(const clang::ArraySubscriptExpr* array);
    bool analyzeInitListExpression(const clang::InitListExpr* initList);
    
    // Context-aware analysis
    bool isInDeterministicContext() const;
    void pushDeterministicContext();
    void popDeterministicContext();
    
    // Expression dependency tracking
    std::set<std::string> getDependentVariables(const clang::Expr* expr);
    bool areVariablesDeterministic(const std::set<std::string>& variables);
    
    // Helper methods from base class
    bool isDeterministicIntrinsicCall(const clang::CallExpr* call);
    bool isDeterministicBinaryOp(const clang::BinaryOperator* op);
    bool isDeterministicUnaryOp(const clang::UnaryOperator* op);
    bool isDeterministicDeclRef(const clang::DeclRefExpr* ref);
    bool isDeterministicMemberAccess(const clang::MemberExpr* member);
    
    // Core data members
    clang::ASTContext& context_;
    std::vector<bool> deterministicContextStack_;
    std::map<std::string, bool> variableDeterminismCache_;
};

// Standalone control flow analyzer
class ControlFlowAnalyzer {
public:
    // Control flow state structure
    struct ControlFlowState {
        bool isDeterministic = true;
        bool hasDeterministicConditions = true;
        bool hasWaveOps = false;
        int deterministicNestingLevel = 0;
    };

    explicit ControlFlowAnalyzer(clang::ASTContext& context)
        : context_(context), deterministicAnalyzer_(context) {}
    
    // Complete implementations of all control flow validation
    ValidationResult analyzeFunction(clang::FunctionDecl* func);
    ValidationResult analyzeStatement(const clang::Stmt* stmt, ControlFlowState& state);
    
    // Complete control flow construct validation
    ValidationResult validateDeterministicIf(const clang::IfStmt* ifStmt, ControlFlowState& state);
    ValidationResult validateDeterministicFor(const clang::ForStmt* forStmt, ControlFlowState& state);
    ValidationResult validateDeterministicWhile(const clang::WhileStmt* whileStmt, ControlFlowState& state);
    ValidationResult validateDeterministicSwitch(const clang::SwitchStmt* switchStmt, ControlFlowState& state);
    
private:
    clang::ASTContext& context_;
    DeterministicExpressionAnalyzer deterministicAnalyzer_;
    
    // Advanced control flow analysis
    ValidationResult analyzeNestedControlFlow(const clang::Stmt* stmt, ControlFlowState& state);
    ValidationResult validateLoopTermination(const clang::Expr* condition, const clang::Expr* increment);
    ValidationResult validateSwitchCases(const clang::SwitchStmt* switchStmt, ControlFlowState& state);
    ValidationResult analyzeBreakContinueFlow(const clang::Stmt* stmt, ControlFlowState& state);
    
    // Control flow pattern recognition
    bool isSimpleDeterministicLoop(const clang::ForStmt* forStmt);
    bool isCountBasedLoop(const clang::ForStmt* forStmt);
    bool isLaneIndexBasedBranch(const clang::IfStmt* ifStmt);
    
    // Flow analysis utilities
    void updateControlFlowState(ControlFlowState& state, const clang::Stmt* stmt);
    bool checkControlFlowConsistency(const ControlFlowState& state);
    ValidationResult mergeControlFlowResults(const std::vector<ValidationResult>& results);
};

// Standalone memory safety analyzer
class MemorySafetyAnalyzer {
public:
    // Memory operation structure
    struct MemoryOperation {
        std::string resourceName;
        clang::SourceLocation location;
        bool isWrite;
        bool isAtomic;
        std::string operationType;
        const clang::Expr* addressExpr;
    };

    // Memory access pattern enumeration
    enum class MemoryAccessPattern {
        LaneDisjoint,      // Each lane accesses different memory
        WaveShared,        // Wave-level shared access
        ThreadgroupShared, // Threadgroup-level shared access
        GlobalShared,      // Global shared access
        Unknown            // Cannot determine pattern
    };

    explicit MemorySafetyAnalyzer(clang::ASTContext& context)
        : context_(context) {}
    
    // Complete memory safety analysis
    ValidationResult analyzeFunction(clang::FunctionDecl* func);
    void collectMemoryOperations(clang::FunctionDecl* func);
    bool hasDisjointWrites();
    bool hasOnlyCommutativeOperations();
    bool hasMemoryRaceCondition(const MemoryOperation& op1, const MemoryOperation& op2);
    
private:
    // Advanced memory analysis
    ValidationResult performAliasAnalysis();
    ValidationResult analyzeSharedMemoryUsage();
    ValidationResult validateAtomicOperations();
    ValidationResult checkMemoryAccessPatterns();
    
    // Sophisticated alias analysis
    bool couldAlias(const clang::Expr* addr1, const clang::Expr* addr2);
    bool analyzeAddressExpressions(const clang::Expr* addr1, const clang::Expr* addr2);
    bool isDisjointLaneBasedAccess(const clang::Expr* addr1, const clang::Expr* addr2);
    bool isConstantOffset(const clang::Expr* expr, int64_t& offset);
    
    // Memory operation classification  
    MemoryAccessPattern classifyMemoryAccess(const MemoryOperation& op);
    bool isCommutativeMemoryOperation(const MemoryOperation& op);
    bool requiresAtomicity(const MemoryOperation& op);
    
    // Data flow analysis
    std::set<std::string> getAliasSet(const std::string& resourceName);
    void buildMemoryDependencyGraph();
    ValidationResult analyzeMemoryDependencies();
    
    clang::ASTContext& context_;
    std::vector<MemoryOperation> memoryOperations_;
    std::map<std::string, MemoryAccessPattern> memoryAccessPatterns_;
    std::map<std::pair<std::string, std::string>, bool> aliasCache_;
};

// Standalone wave operation validator
class WaveOperationValidator {
public:
    // Wave operation type enumeration
    enum class WaveOperationType {
        Reduction,         // WaveActiveSum, WaveActiveMax, etc.
        Broadcast,         // WaveReadLaneAt, etc. (forbidden)
        Query,             // WaveGetLaneIndex, WaveGetLaneCount
        Ballot,            // WaveBallot, etc. (forbidden)
        Prefix,            // WavePrefixSum, etc. (forbidden)
        Unknown
    };

    explicit WaveOperationValidator(clang::ASTContext& context)
        : context_(context) {}
    
    // Complete wave operation validation
    ValidationResult validateWaveCall(const clang::CallExpr* call, 
                                     const ControlFlowAnalyzer::ControlFlowState& cfState);
    bool isWaveOperation(const clang::CallExpr* call);
    bool isOrderIndependentWaveOp(const std::string& funcName);
    bool isOrderDependentWaveOp(const std::string& funcName);
    bool requiresFullParticipation(const std::string& funcName);
    
private:
    // Advanced wave operation analysis
    ValidationResult analyzeWaveParticipation(const clang::CallExpr* call, 
                                             const ControlFlowAnalyzer::ControlFlowState& cfState);
    ValidationResult validateWaveArguments(const clang::CallExpr* call, const std::string& funcName);
    ValidationResult checkWaveOperationContext(const clang::CallExpr* call);
    
    // Wave operation classification
    WaveOperationType classifyWaveOperation(const std::string& funcName);
    bool isReductionOperation(const std::string& funcName);
    bool isBroadcastOperation(const std::string& funcName);
    bool isQueryOperation(const std::string& funcName);
    
    // Context analysis
    bool isInUniformControlFlow(const ControlFlowAnalyzer::ControlFlowState& cfState);
    bool isInDivergentControlFlow(const ControlFlowAnalyzer::ControlFlowState& cfState);
    int calculateDivergenceLevel(const ControlFlowAnalyzer::ControlFlowState& cfState);
    
    clang::ASTContext& context_;
    std::map<std::string, WaveOperationType> waveOperationTypes_;
};

// Main HLSL validator with full AST implementation
class MiniHLSLValidator {
public:
    MiniHLSLValidator();
    ~MiniHLSLValidator() = default;
    
    // Main validation methods
    ValidationResult validateProgram(const Program* program);
    ValidationResult validateFunction(const Function* func);
    ValidationResult validateSource(const std::string& hlslSource);
    
    // Additional AST-based methods
    ValidationResult validateAST(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult validateSourceWithFullAST(const std::string& hlslSource, 
                                               const std::string& filename = "shader.hlsl");
    
    // Complete formal proof integration
    std::string generateFormalProofAlignment(const Program* program);
    FormalProofMapping mapToFormalProof(const ValidationResult& result, const Program* program);
    
    // Advanced analysis capabilities
    ValidationResult performFullStaticAnalysis(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult analyzeOrderIndependence(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult generateOptimizationSuggestions(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    
private:
    // Analyzer instances
    std::unique_ptr<DeterministicExpressionAnalyzer> expressionAnalyzer_;
    std::unique_ptr<ControlFlowAnalyzer> controlFlowAnalyzer_;
    std::unique_ptr<MemorySafetyAnalyzer> memoryAnalyzer_;
    std::unique_ptr<WaveOperationValidator> waveValidator_;
    
    // AST parsing and setup
    std::pair<std::unique_ptr<clang::ASTContext>, clang::TranslationUnitDecl*> 
    parseHLSLWithCompleteAST(const std::string& source, const std::string& filename);
    
    std::unique_ptr<clang::CompilerInstance> setupCompleteCompiler();
    
    // Complete validation workflow
    ValidationResult runCompleteValidation(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult validateAllConstraints(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult generateComprehensiveReport(const ValidationResult& result, 
                                                clang::TranslationUnitDecl* tu, 
                                                clang::ASTContext& context);
    
    // Analysis coordination
    void initializeAnalyzers(clang::ASTContext& context);
    ValidationResult coordinateAnalysis(clang::TranslationUnitDecl* tu, clang::ASTContext& context);
    ValidationResult consolidateResults(const std::vector<ValidationResult>& results);
};

// Factory for creating validators
class ValidatorFactory {
public:
    static std::unique_ptr<MiniHLSLValidator> createValidator();
    static std::unique_ptr<MiniHLSLValidator> createMiniHLSLValidator();
    
    // Create validator with specific analyzers
    static std::unique_ptr<MiniHLSLValidator> createValidatorWithAnalyzers(
        std::unique_ptr<DeterministicExpressionAnalyzer> expressionAnalyzer,
        std::unique_ptr<ControlFlowAnalyzer> controlFlowAnalyzer,
        std::unique_ptr<MemorySafetyAnalyzer> memoryAnalyzer,
        std::unique_ptr<WaveOperationValidator> waveValidator);
};

} // namespace minihlsl