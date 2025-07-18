#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <optional>

namespace minihlsl {

// Forward declarations for AST nodes
struct Expression;
struct Statement;
struct Function;
struct Program;

// Validation result types
enum class ValidationError {
    // Control flow violations
    NonUniformBranch,
    WaveDivergentLoop,
    MissingReconvergence,
    
    // Wave operation violations  
    OrderDependentWaveOp,
    IncompleteWaveParticipation,
    InvalidWaveContext,
    
    // Memory access violations
    UnsynchronizedWrite,
    DataRace,
    NonUniformMemoryAccess,
    
    // Type/syntax violations
    UnsupportedType,
    UnsupportedOperation,
    InvalidExpression,
    
    // General violations
    NonDeterministicOperation,
    SideEffectInPureFunction,
    UniformityViolation
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
};

// Uniformity analysis - tracks which expressions are uniform across wave
class UniformityAnalyzer {
public:
    UniformityAnalyzer() = default;
    
    // Check if expression is uniform (same value across all lanes)
    bool isUniform(const Expression* expr) const;
    
    // Mark variable as uniform
    void markUniform(const std::string& varName);
    
    // Mark variable as divergent (may vary per lane)  
    void markDivergent(const std::string& varName);
    
    // Analyze expression and update uniformity state
    void analyzeExpression(const Expression* expr);
    
private:
    std::unordered_set<std::string> uniformVariables;
    std::unordered_set<std::string> divergentVariables;
    
    // Built-in uniform intrinsics
    static const std::unordered_set<std::string> uniformIntrinsics;
    
    // Built-in divergent intrinsics  
    static const std::unordered_set<std::string> divergentIntrinsics;
};

// Control flow analyzer - ensures proper wave convergence
class ControlFlowAnalyzer {
public:
    struct ControlFlowState {
        bool isConverged = true;        // All lanes follow same path
        bool hasWaveOps = false;        // Contains wave operations
        bool hasUniformBranch = true;   // Branch condition is uniform
        int divergenceLevel = 0;        // Nesting level of divergent branches
    };
    
    ValidationResult analyzeFunction(const Function* func);
    
private:
    ValidationResult analyzeStatement(const Statement* stmt, ControlFlowState& state);
    ValidationResult analyzeBlock(const std::vector<Statement*>& stmts, ControlFlowState& state);
    
    // Check if condition is uniform across wave
    bool isUniformCondition(const Expression* condition) const;
    
    UniformityAnalyzer uniformityAnalyzer;
};

// Wave operation validator - ensures correct wave intrinsic usage
class WaveOperationValidator {
public:
    ValidationResult validateWaveOp(const std::string& intrinsicName, 
                                   const std::vector<Expression*>& args,
                                   const ControlFlowAnalyzer::ControlFlowState& cfState);
    
private:
    // Order-independent wave operations (allowed)
    static const std::unordered_set<std::string> orderIndependentOps;
    
    // Order-dependent wave operations (forbidden)
    static const std::unordered_set<std::string> orderDependentOps;
    
    // Operations requiring full wave participation
    static const std::unordered_set<std::string> fullParticipationOps;
    
    bool requiresFullParticipation(const std::string& intrinsicName) const;
    bool isOrderDependent(const std::string& intrinsicName) const;
};

// Memory access analyzer - detects potential data races
class MemoryAccessAnalyzer {
public:
    struct MemoryAccess {
        std::string resourceName;
        bool isWrite;
        bool isUniform;  // Access pattern is uniform across wave
        const Expression* indexExpression;
    };
    
    ValidationResult analyzeMemoryAccesses(const Function* func);
    
private:
    std::vector<MemoryAccess> memoryAccesses;
    
    bool hasDataRace(const MemoryAccess& access1, const MemoryAccess& access2) const;
    void collectMemoryAccesses(const Statement* stmt);
    void collectFromExpression(const Expression* expr);
};

// Main validator class
class MiniHLSLValidator {
public:
    MiniHLSLValidator() = default;
    
    // Validate complete program for order-independence
    ValidationResult validateProgram(const Program* program);
    
    // Validate individual function
    ValidationResult validateFunction(const Function* func);
    
    // Quick validation of HLSL source string
    ValidationResult validateSource(const std::string& hlslSource);
    
    // Check if specific construct is allowed in MiniHLSL
    bool isAllowedConstruct(const std::string& constructName) const;
    
private:
    ControlFlowAnalyzer controlFlowAnalyzer;
    WaveOperationValidator waveValidator;  
    MemoryAccessAnalyzer memoryAnalyzer;
    
    // Validate function signature for MiniHLSL compliance
    ValidationResult validateFunctionSignature(const Function* func);
    
    // Validate variable declarations
    ValidationResult validateVariableDecl(const Statement* stmt);
    
    // Check for forbidden language constructs
    ValidationResult checkForbiddenConstructs(const Program* program);
    
    // Forbidden constructs in MiniHLSL
    static const std::unordered_set<std::string> forbiddenKeywords;
    static const std::unordered_set<std::string> forbiddenIntrinsics;
};

// Utility functions for order-independence verification

// Check if operation is commutative (order doesn't matter)
bool isCommutativeOperation(const std::string& op);

// Check if operation is associative (grouping doesn't matter)  
bool isAssociativeOperation(const std::string& op);

// Check if expression is deterministic (same inputs -> same output)
bool isDeterministicExpression(const Expression* expr);

// Verify that all execution paths reconverge before wave operations
bool verifyReconvergence(const Function* func);

// Generate order-independent test variants for fuzzing
std::vector<std::string> generateOrderIndependentVariants(const std::string& baseProgram);

} // namespace minihlsl