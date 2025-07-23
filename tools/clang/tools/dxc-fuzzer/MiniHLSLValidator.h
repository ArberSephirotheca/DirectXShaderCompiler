#pragma once

// Core validation types and enums
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace minihlsl {

// Type aliases for cleaner API (Rust-style naming)
using Expression = clang::Expr;
using Statement = clang::Stmt;
using Function = clang::FunctionDecl;
using Program = clang::TranslationUnitDecl;

// Rust-style Result and Option types for C++
template <typename T, typename E> class Result {
public:
  static Result Ok(T value) { return Result(std::move(value), true); }
  static Result Err(E error) { return Result(std::move(error), false); }

  bool is_ok() const { return is_success_; }
  bool is_err() const { return !is_success_; }

  const T &unwrap() const {
    if (!is_success_)
      throw std::runtime_error("Called unwrap on error Result");
    return std::get<T>(data_);
  }

  const E &unwrap_err() const {
    if (is_success_)
      throw std::runtime_error("Called unwrap_err on ok Result");
    return std::get<E>(data_);
  }

  template <typename F>
  auto map(F &&func) -> Result<decltype(func(unwrap())), E> {
    if (is_ok()) {
      return Result<decltype(func(unwrap())), E>::Ok(func(unwrap()));
    }
    return Result<decltype(func(unwrap())), E>::Err(unwrap_err());
  }

  template <typename F> auto and_then(F &&func) -> decltype(func(unwrap())) {
    if (is_ok()) {
      return func(unwrap());
    }
    return decltype(func(unwrap()))::Err(unwrap_err());
  }

private:
  std::variant<T, E> data_;
  bool is_success_;

  Result(T value, bool) : data_(std::move(value)), is_success_(true) {}
  Result(E error, bool) : data_(std::move(error)), is_success_(false) {}
};

template <typename T> using Option = std::optional<T>;

// Rust-style smart pointer aliases
template <typename T> using Box = std::unique_ptr<T>;

template <typename T> using Rc = std::shared_ptr<T>;

// Helper functions for Rust-style construction
template <typename T, typename... Args> Box<T> make_box(Args &&...args) {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T> Option<T> Some(T value) {
  return std::make_optional(std::move(value));
}

template <typename T> Option<T> None() { return std::nullopt; }

// Rust-style error enum with associated data
enum class ValidationError {
  // Deterministic control flow violations
  NonDeterministicCondition,      // Control flow condition not compile-time
                                  // deterministic
  InvalidDeterministicExpression, // Expression not compile-time deterministic
  MixedDeterministicContext,      // Mixing deterministic and non-deterministic
                                  // constructs

  // Wave operation violations
  OrderDependentWaveOp,        // Use of order-dependent operations (forbidden)
  IncompleteWaveParticipation, // Wave operation in divergent context
  InvalidWaveContext,          // Wave operation in invalid control flow context

  // Memory access violations
  OverlappingMemoryWrites,   // Violation of hasDisjointWrites constraint
  NonCommutativeMemoryOp,    // Violation of hasOnlyCommutativeOps constraint
  SharedMemoryRaceCondition, // Race condition in shared memory access
  DataRaceCondition,         // Violation of data-race-free memory model
  UnsynchronizedCompoundRMW, // Compound read-modify-write without atomics or
                             // barriers
  NonAtomicRMWSameThread,    // Same-thread RMW to same address without atomics

  // Type/syntax violations
  UnsupportedType,
  UnsupportedOperation,
  InvalidExpression,

  // General violations
  NonDeterministicOperation,
  SideEffectInPureFunction,
  ForbiddenLanguageConstruct
};

// Rust-style error with message
struct ValidationErrorWithMessage {
  ValidationError kind;
  std::string message;
  Option<clang::SourceLocation> location;

  ValidationErrorWithMessage(
      ValidationError k, std::string msg,
      Option<clang::SourceLocation> loc = None<clang::SourceLocation>())
      : kind(k), message(std::move(msg)), location(loc) {}
};

// Rust-style Result type for validation
using ValidationResult =
    Result<std::monostate, std::vector<ValidationErrorWithMessage>>;

// Helper functions for creating results
class ValidationResultBuilder {
public:
  static ValidationResult ok() {
    return ValidationResult::Ok(std::monostate{});
  }

  static ValidationResult
  err(ValidationError error, const std::string &message,
      Option<clang::SourceLocation> location = None<clang::SourceLocation>()) {
    std::vector<ValidationErrorWithMessage> errors;
    errors.emplace_back(error, message, location);
    return ValidationResult::Err(std::move(errors));
  }

  static ValidationResult
  combine(const std::vector<ValidationResult> &results) {
    std::vector<ValidationErrorWithMessage> all_errors;

    for (const auto &result : results) {
      if (result.is_err()) {
        const auto &errors = result.unwrap_err();
        all_errors.insert(all_errors.end(), errors.begin(), errors.end());
      }
    }

    if (all_errors.empty()) {
      return ok();
    }
    return ValidationResult::Err(std::move(all_errors));
  }
};

// Formal verification integration utilities
struct FormalProofMapping {
  bool deterministicControlFlow;   // Maps to hasDetministicControlFlow
  bool orderIndependentOperations; // Maps to wave/threadgroup operation
                                   // constraints
  bool memorySafetyConstraints;    // Maps to hasDisjointWrites &
                                   // hasOnlyCommutativeOps
  bool programOrderIndependence;   // Overall conclusion
};

// Forward declarations
class DeterministicExpressionAnalyzer;
class ControlFlowAnalyzer;
class MemorySafetyAnalyzer;
class WaveOperationValidator;
class HybridRMWAnalyzer;

// ASTConsumer for DBEG analysis that plugs into DXC compilation
class DBEGAnalysisConsumer : public clang::ASTConsumer, 
                             public clang::RecursiveASTVisitor<DBEGAnalysisConsumer> {
private:
  std::unique_ptr<MemorySafetyAnalyzer> memory_analyzer_;
  std::vector<ValidationErrorWithMessage> errors_;
  ValidationResult final_result_;
  bool analysis_completed_;
  
public:
  explicit DBEGAnalysisConsumer() 
    : memory_analyzer_(nullptr), final_result_(ValidationResultBuilder::ok()), analysis_completed_(false) {
    llvm::errs() << "DBEGAnalysisConsumer constructor called\n";
  }
  
  // ASTConsumer interface - this is where ALL analysis happens
  void HandleTranslationUnit(clang::ASTContext &context) override;
  
  // RecursiveASTVisitor interface
  bool VisitFunctionDecl(clang::FunctionDecl *func);
  
private:
  // Helper function to validate wave operations in a function
  ValidationResult validate_wave_operations_in_function(
      clang::FunctionDecl *func, const WaveOperationValidator &validator);
      
  // Helper function to validate deterministic expressions in a function
  ValidationResult validate_deterministic_expressions_in_function(
      clang::FunctionDecl *func, const DeterministicExpressionAnalyzer &analyzer);
      
  // Helper function to validate control flow in a function
  ValidationResult validate_control_flow_in_function(
      clang::FunctionDecl *func, const ControlFlowAnalyzer &analyzer);
      
  // Helper function to validate memory safety patterns in a function
  ValidationResult validate_memory_safety_patterns_in_function(clang::FunctionDecl *func);
      
public:
  // Get analysis results - only valid after HandleTranslationUnit completes
  const ValidationResult& getFinalResult() const { return final_result_; }
  const std::vector<ValidationErrorWithMessage>& getErrors() const { return errors_; }
  bool hasErrors() const { return !errors_.empty(); }
  bool isAnalysisCompleted() const { return analysis_completed_; }
};

// FrontendAction to create our ASTConsumer
class DBEGAnalysisAction : public clang::ASTFrontendAction {
private:
  DBEGAnalysisConsumer* consumer_; // Raw pointer to track consumer - only valid during execution
  
  // Copy results before consumer is deleted
  ValidationResult final_result_;
  std::vector<ValidationErrorWithMessage> captured_errors_;
  bool analysis_completed_;
  bool results_captured_;
  
public:
  DBEGAnalysisAction() : consumer_(nullptr), final_result_(ValidationResultBuilder::ok()), 
                         analysis_completed_(false), results_captured_(false) {}
  
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &CI, clang::StringRef InFile) override;
      
  // Override to capture results before consumer is deleted
  void ExecuteAction() override;
      
  // Get analysis results after compilation - uses captured results
  const ValidationResult& getFinalResult() const {
    static ValidationResult default_success = ValidationResultBuilder::ok();
    return results_captured_ ? final_result_ : default_success;
  }
  bool hasAnalysisErrors() const {
    return results_captured_ ? !captured_errors_.empty() : false;
  }
  bool isAnalysisCompleted() const {
    return results_captured_ ? analysis_completed_ : false;
  }
};

// Production-ready implementation of all AST-based validation components
// This provides complete implementations of all validation methods

// Standalone deterministic expression analyzer
class DeterministicExpressionAnalyzer {
public:
  enum class ExpressionKind {
    Literal,         // Constants: 42, 3.14f, true
    LaneIndex,       // WaveGetLaneIndex()
    WaveIndex,       // Current wave ID
    ThreadIndex,     // Global thread index
    WaveProperty,    // WaveGetLaneCount(), etc.
    Arithmetic,      // +, -, *, / of deterministic expressions
    Comparison,      // <, >, ==, != of deterministic expressions
    NonDeterministic // Everything else
  };

  explicit DeterministicExpressionAnalyzer(clang::ASTContext &context)
      : context_(context) {}

  // Rust-style constructor
  static Box<DeterministicExpressionAnalyzer>
  create(clang::ASTContext &context) {
    return make_box<DeterministicExpressionAnalyzer>(context);
  }

  // Core implementations with Rust-style naming and return types
  bool is_compile_time_deterministic(const clang::Expr *expr);
  ExpressionKind classify_expression(const clang::Expr *expr);
  ValidationResult validate_deterministic_expression(const clang::Expr *expr);

  // Legacy C++ style methods for compatibility
  bool isCompileTimeDeterministic(const clang::Expr *expr) {
    return is_compile_time_deterministic(expr);
  }
  ExpressionKind classifyExpression(const clang::Expr *expr) {
    return classify_expression(expr);
  }
  ValidationResult validateDeterministicExpression(const clang::Expr *expr) {
    return validate_deterministic_expression(expr);
  }

  // Rust-style helper methods
  bool is_literal_constant(const clang::Expr *expr);
  bool is_lane_index_expression(const clang::Expr *expr);
  bool is_wave_property_expression(const clang::Expr *expr);
  bool is_thread_index_expression(const clang::Expr *expr);
  bool is_arithmetic_of_deterministic(const clang::Expr *expr);
  bool is_comparison_of_deterministic(const clang::Expr *expr);

  // Legacy compatibility methods
  bool isLiteralConstant(const clang::Expr *expr) {
    return is_literal_constant(expr);
  }
  bool isLaneIndexExpression(const clang::Expr *expr) {
    return is_lane_index_expression(expr);
  }
  bool isWavePropertyExpression(const clang::Expr *expr) {
    return is_wave_property_expression(expr);
  }
  bool isThreadIndexExpression(const clang::Expr *expr) {
    return is_thread_index_expression(expr);
  }
  bool isArithmeticOfDeterministic(const clang::Expr *expr) {
    return is_arithmetic_of_deterministic(expr);
  }
  bool isComparisonOfDeterministic(const clang::Expr *expr) {
    return is_comparison_of_deterministic(expr);
  }

private:
  // Rust-style advanced expression analysis methods
  bool analyze_complex_expression(const clang::Expr *expr);
  bool analyze_conditional_operator(const clang::ConditionalOperator *cond);
  bool analyze_cast_expression(const clang::CastExpr *cast);
  bool analyze_array_subscript(const clang::ArraySubscriptExpr *array);
  bool analyze_init_list_expression(const clang::InitListExpr *init_list);

  // Legacy compatibility methods
  bool analyzeComplexExpression(const clang::Expr *expr) {
    return analyze_complex_expression(expr);
  }
  bool analyzeConditionalOperator(const clang::ConditionalOperator *cond) {
    return analyze_conditional_operator(cond);
  }
  bool analyzeCastExpression(const clang::CastExpr *cast) {
    return analyze_cast_expression(cast);
  }
  bool analyzeArraySubscript(const clang::ArraySubscriptExpr *array) {
    return analyze_array_subscript(array);
  }
  bool analyzeInitListExpression(const clang::InitListExpr *initList) {
    return analyze_init_list_expression(initList);
  }

  // Rust-style context-aware analysis
  bool is_in_deterministic_context() const;
  void push_deterministic_context();
  void pop_deterministic_context();

  // Legacy compatibility
  bool isInDeterministicContext() const {
    return is_in_deterministic_context();
  }
  void pushDeterministicContext() { push_deterministic_context(); }
  void popDeterministicContext() { pop_deterministic_context(); }

  // Rust-style expression dependency tracking
  std::set<std::string> get_dependent_variables(const clang::Expr *expr);
  bool are_variables_deterministic(const std::set<std::string> &variables);

  // Legacy compatibility
  std::set<std::string> getDependentVariables(const clang::Expr *expr) {
    return get_dependent_variables(expr);
  }
  bool areVariablesDeterministic(const std::set<std::string> &variables) {
    return are_variables_deterministic(variables);
  }

  // Rust-style helper methods
  bool is_deterministic_intrinsic_call(const clang::CallExpr *call);
  bool is_deterministic_binary_op(const clang::BinaryOperator *op);
  bool is_deterministic_unary_op(const clang::UnaryOperator *op);
  bool is_deterministic_decl_ref(const clang::DeclRefExpr *ref);
  bool is_deterministic_member_access(const clang::MemberExpr *member);
  bool is_deterministic_member_call(const clang::CXXMemberCallExpr *call);

  // Legacy compatibility
  bool isDeterministicIntrinsicCall(const clang::CallExpr *call) {
    return is_deterministic_intrinsic_call(call);
  }
  bool isDeterministicBinaryOp(const clang::BinaryOperator *op) {
    return is_deterministic_binary_op(op);
  }
  bool isDeterministicUnaryOp(const clang::UnaryOperator *op) {
    return is_deterministic_unary_op(op);
  }
  bool isDeterministicDeclRef(const clang::DeclRefExpr *ref) {
    return is_deterministic_decl_ref(ref);
  }
  bool isDeterministicMemberAccess(const clang::MemberExpr *member) {
    return is_deterministic_member_access(member);
  }

  // Rust-style data members
  clang::ASTContext &context_;
  std::vector<bool> deterministic_context_stack_;
  std::map<std::string, bool> variable_determinism_cache_;

  // Legacy compatibility members
  std::vector<bool> &deterministicContextStack_ = deterministic_context_stack_;
  std::map<std::string, bool> &variableDeterminismCache_ =
      variable_determinism_cache_;
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

  explicit ControlFlowAnalyzer(clang::ASTContext &context)
      : context_(context), deterministic_analyzer_(context) {}

  // Rust-style constructor
  static Box<ControlFlowAnalyzer> create(clang::ASTContext &context) {
    return make_box<ControlFlowAnalyzer>(context);
  }

  // Rust-style method names
  ValidationResult analyze_function(clang::FunctionDecl *func);
  ValidationResult analyze_statement(const clang::Stmt *stmt,
                                     ControlFlowState &state);

  // Legacy compatibility
  ValidationResult analyzeFunction(clang::FunctionDecl *func) {
    return analyze_function(func);
  }
  ValidationResult analyzeStatement(const clang::Stmt *stmt,
                                    ControlFlowState &state) {
    return analyze_statement(stmt, state);
  }

  // Rust-style control flow construct validation
  ValidationResult validate_deterministic_if(const clang::IfStmt *if_stmt,
                                             ControlFlowState &state);
  ValidationResult validate_deterministic_for(const clang::ForStmt *for_stmt,
                                              ControlFlowState &state);
  ValidationResult
  validate_deterministic_while(const clang::WhileStmt *while_stmt,
                               ControlFlowState &state);
  ValidationResult
  validate_deterministic_switch(const clang::SwitchStmt *switch_stmt,
                                ControlFlowState &state);

  // Legacy compatibility
  ValidationResult validateDeterministicIf(const clang::IfStmt *ifStmt,
                                           ControlFlowState &state) {
    return validate_deterministic_if(ifStmt, state);
  }
  ValidationResult validateDeterministicFor(const clang::ForStmt *forStmt,
                                            ControlFlowState &state) {
    return validate_deterministic_for(forStmt, state);
  }
  ValidationResult validateDeterministicWhile(const clang::WhileStmt *whileStmt,
                                              ControlFlowState &state) {
    return validate_deterministic_while(whileStmt, state);
  }
  ValidationResult
  validateDeterministicSwitch(const clang::SwitchStmt *switchStmt,
                              ControlFlowState &state) {
    return validate_deterministic_switch(switchStmt, state);
  }

private:
  clang::ASTContext &context_;
  DeterministicExpressionAnalyzer deterministic_analyzer_;

  // Rust-style advanced control flow analysis
  ValidationResult analyze_nested_control_flow(const clang::Stmt *stmt,
                                               ControlFlowState &state);
  ValidationResult validate_loop_termination(const clang::Expr *condition,
                                             const clang::Expr *increment);
  ValidationResult validate_switch_cases(const clang::SwitchStmt *switch_stmt,
                                         ControlFlowState &state);
  ValidationResult analyze_break_continue_flow(const clang::Stmt *stmt,
                                               ControlFlowState &state);

  // Legacy compatibility
  ValidationResult analyzeNestedControlFlow(const clang::Stmt *stmt,
                                            ControlFlowState &state) {
    return analyze_nested_control_flow(stmt, state);
  }
  ValidationResult validateLoopTermination(const clang::Expr *condition,
                                           const clang::Expr *increment) {
    return validate_loop_termination(condition, increment);
  }
  ValidationResult validateSwitchCases(const clang::SwitchStmt *switchStmt,
                                       ControlFlowState &state) {
    return validate_switch_cases(switchStmt, state);
  }
  ValidationResult analyzeBreakContinueFlow(const clang::Stmt *stmt,
                                            ControlFlowState &state) {
    return analyze_break_continue_flow(stmt, state);
  }

  // Rust-style control flow pattern recognition
  bool is_simple_deterministic_loop(const clang::ForStmt *for_stmt);
  bool is_count_based_loop(const clang::ForStmt *for_stmt);
  bool is_lane_index_based_branch(const clang::IfStmt *if_stmt);
  bool is_simple_count_loop(const clang::Expr *condition, const clang::Expr *increment);

  // Legacy compatibility
  bool isSimpleDeterministicLoop(const clang::ForStmt *forStmt) {
    return is_simple_deterministic_loop(forStmt);
  }
  bool isCountBasedLoop(const clang::ForStmt *forStmt) {
    return is_count_based_loop(forStmt);
  }
  bool isLaneIndexBasedBranch(const clang::IfStmt *ifStmt) {
    return is_lane_index_based_branch(ifStmt);
  }

  // Rust-style flow analysis utilities
  void update_control_flow_state(ControlFlowState &state,
                                 const clang::Stmt *stmt);
  bool check_control_flow_consistency(const ControlFlowState &state);
  ValidationResult
  merge_control_flow_results(const std::vector<ValidationResult> &results);

  // Legacy compatibility
  void updateControlFlowState(ControlFlowState &state,
                              const clang::Stmt *stmt) {
    update_control_flow_state(state, stmt);
  }
  bool checkControlFlowConsistency(const ControlFlowState &state) {
    return check_control_flow_consistency(state);
  }
  ValidationResult
  mergeControlFlowResults(const std::vector<ValidationResult> &results) {
    return merge_control_flow_results(results);
  }
};

// Standalone memory safety analyzer
class MemorySafetyAnalyzer {
public:
  // Memory operation structure
  struct MemoryOperation {
    std::string resourceName;
    clang::SourceLocation location;
    bool isWrite;
    bool isRead;
    bool isAtomic;
    std::string operationType;
    const clang::Expr *addressExpr;
    int threadId;
    bool isSameThreadRMW;
    bool isSynchronized;
  };

  // Memory access pattern enumeration
  enum class MemoryAccessPattern {
    LaneDisjoint,      // Each lane accesses different memory
    WaveShared,        // Wave-level shared access
    ThreadgroupShared, // Threadgroup-level shared access
    GlobalShared,      // Global shared access
    Unknown            // Cannot determine pattern
  };

  explicit MemorySafetyAnalyzer(clang::ASTContext &context)
      : context_(context) {}

  // Rust-style constructor
  static Box<MemorySafetyAnalyzer> create(clang::ASTContext &context) {
    return make_box<MemorySafetyAnalyzer>(context);
  }

  // Rust-style method names
  ValidationResult analyze_function(clang::FunctionDecl *func);
  void collect_memory_operations(clang::FunctionDecl *func);
  bool has_disjoint_writes();
  bool has_only_commutative_operations();
  bool has_memory_race_condition(const MemoryOperation &op1,
                                 const MemoryOperation &op2);

  // Legacy compatibility
  ValidationResult analyzeFunction(clang::FunctionDecl *func) {
    return analyze_function(func);
  }
  void collectMemoryOperations(clang::FunctionDecl *func) {
    collect_memory_operations(func);
  }
  bool hasDisjointWrites() { return has_disjoint_writes(); }
  bool hasOnlyCommutativeOperations() {
    return has_only_commutative_operations();
  }
  bool hasMemoryRaceCondition(const MemoryOperation &op1,
                              const MemoryOperation &op2) {
    return has_memory_race_condition(op1, op2);
  }

  // Rust-style data-race-free memory model validation
  ValidationResult validate_data_race_freedom();
  bool has_data_race(const MemoryOperation &op1, const MemoryOperation &op2);
  bool are_conflicting_accesses(const MemoryOperation &op1,
                                const MemoryOperation &op2);
  bool are_synchronized(const MemoryOperation &op1, const MemoryOperation &op2);

  // Rust-style hybrid approach for compound RMW operations
  ValidationResult validate_hybrid_rmw_approach();
  bool is_simple_rmw_operation(const MemoryOperation &read,
                               const MemoryOperation &write);
  bool requires_atomic_rmw(const MemoryOperation &read,
                           const MemoryOperation &write);
  bool has_barrier_synchronization(const MemoryOperation &op1,
                                   const MemoryOperation &op2);

  // Legacy compatibility methods
  ValidationResult validateDataRaceFreedom() {
    return validate_data_race_freedom();
  }
  bool hasDataRace(const MemoryOperation &op1, const MemoryOperation &op2) {
    return has_data_race(op1, op2);
  }
  bool areConflictingAccesses(const MemoryOperation &op1,
                              const MemoryOperation &op2) {
    return are_conflicting_accesses(op1, op2);
  }
  bool areSynchronized(const MemoryOperation &op1, const MemoryOperation &op2) {
    return are_synchronized(op1, op2);
  }
  ValidationResult validateHybridRMWApproach() {
    return validate_hybrid_rmw_approach();
  }
  bool isSimpleRMWOperation(const MemoryOperation &read,
                            const MemoryOperation &write) {
    return is_simple_rmw_operation(read, write);
  }
  bool requiresAtomicRMW(const MemoryOperation &read,
                         const MemoryOperation &write) {
    return requires_atomic_rmw(read, write);
  }
  bool hasBarrierSynchronization(const MemoryOperation &op1,
                                 const MemoryOperation &op2) {
    return has_barrier_synchronization(op1, op2);
  }

private:
  // Rust-style advanced memory analysis
  ValidationResult perform_alias_analysis();
  ValidationResult analyze_shared_memory_usage();
  ValidationResult validate_atomic_operations();
  ValidationResult check_memory_access_patterns();

  // Legacy compatibility
  ValidationResult performAliasAnalysis() { return perform_alias_analysis(); }
  ValidationResult analyzeSharedMemoryUsage() {
    return analyze_shared_memory_usage();
  }
  ValidationResult validateAtomicOperations() {
    return validate_atomic_operations();
  }
  ValidationResult checkMemoryAccessPatterns() {
    return check_memory_access_patterns();
  }

  // Rust-style sophisticated alias analysis
  bool could_alias(const clang::Expr *addr1, const clang::Expr *addr2);
  bool analyze_address_expressions(const clang::Expr *addr1,
                                   const clang::Expr *addr2);
  bool is_disjoint_lane_based_access(const clang::Expr *addr1,
                                     const clang::Expr *addr2);
  bool is_constant_offset(const clang::Expr *expr, int64_t &offset);
  
  // Basic symbolic expression analysis for array indices
  bool are_indices_definitely_different(const clang::Expr *idx1, const clang::Expr *idx2);
  bool are_same_expression(const clang::Expr *expr1, const clang::Expr *expr2);
  
  // Cross-thread alias analysis
  bool could_alias_across_threads(const clang::Expr *addr1, uint32_t thread1,
                                  const clang::Expr *addr2, uint32_t thread2);
  bool is_thread_id_expression(const clang::Expr *expr);
  const clang::Expr* extract_base_and_offset(const clang::Expr *expr, int64_t &offset);
  bool is_shared_memory_access(const clang::Expr *expr);

  // Dynamic Block Execution Graph (DBEG) for cross-path dependency analysis
  struct DynamicBlock {
    uint32_t id;                              // Unique identifier
    const clang::Stmt* staticBlock;           // Original basic block/statement
    std::set<uint32_t> threads;               // db.T - threads that execute this dynamic block
    int iterationId;                          // Loop iteration (-1 if not in loop)
    const clang::Stmt* parentLoop;           // Parent loop statement (nullptr if not in loop)
    
    // Merge block information (critical for reconvergence)
    uint32_t mergeTargetId;                   // db.merge - where threads reconverge
    std::vector<uint32_t> mergeStack;         // Stack of merge targets for nested control flow
    
    // Graph structure
    std::vector<uint32_t> children;           // Child dynamic blocks
    std::vector<uint32_t> predecessors;       // Predecessor dynamic blocks
    
    // Helper methods
    bool hasThread(uint32_t threadId) const { return threads.find(threadId) != threads.end(); }
    bool isInLoop() const { return iterationId >= 0; }
  };

  struct MemoryOperationInDBEG {
    MemoryOperation op;                       // Original memory operation
    uint32_t dynamicBlockId;                  // Which dynamic block contains this operation
    int programPoint;                         // Execution order within thread
  };

  // DBEG construction and analysis
  std::map<uint32_t, DynamicBlock> dynamicBlocks_;
  std::vector<MemoryOperationInDBEG> dbegMemoryOps_;
  uint32_t nextDynamicBlockId_ = 0;
  
  // Track active threads as we build the DBEG
  std::map<uint32_t, std::set<uint32_t>> activeThreadsAfterBlock_;
  std::set<uint32_t> compute_threads_after_block(uint32_t blockId);
  std::set<uint32_t> remove_exiting_threads(const std::set<uint32_t>& threads, const clang::Stmt* stmt);
  
  // DBEG methods
  void build_dynamic_execution_graph(clang::FunctionDecl *func);
  void build_dynamic_blocks_recursive(
    const clang::Stmt* stmt, 
    uint32_t parentBlockId,
    const std::set<uint32_t>& threads);
  void collect_memory_operations_in_dbeg(
    const clang::Stmt* stmt,
    uint32_t dynamicBlockId,
    const std::set<uint32_t>& activeThreads);
  uint32_t create_dynamic_block(const clang::Stmt* stmt, 
                               const std::set<uint32_t>& threads,
                               int iteration = -1,
                               const clang::Stmt* parentLoop = nullptr);
  std::set<uint32_t> compute_thread_participation(const clang::Stmt* stmt, 
                                                const std::set<uint32_t>& parentThreads);
  std::set<uint32_t> compute_thread_participation_if_branch(
  const clang::IfStmt* if_stmt, const std::set<uint32_t>& threads, bool takeTrueBranch);   
  std::set<uint32_t> compute_threads_executing_iteration(
  const clang::ForStmt* for_stmt, const std::set<uint32_t>& threads, int iteration);
  int compute_max_loop_iterations(const clang::ForStmt* for_stmt, const std::set<uint32_t>& parentThreads);
  bool thread_executes_loop_iteration(const clang::ForStmt* for_stmt, const uint32_t tid, const int iteration);
  const clang::Stmt* find_merge_target_for_if(const clang::IfStmt* if_stmt);
  const clang::Stmt* find_merge_target_for_loop(const clang::ForStmt* for_stmt);
  ValidationResult validate_cross_dynamic_block_dependencies();
  void print_dynamic_execution_graph(bool verbose = false);
  
  // Helper methods for DBEG construction
  const clang::Stmt* get_parent_statement(const clang::Stmt* stmt);
  bool evaluate_deterministic_condition_for_thread(const clang::Expr* condition, uint32_t threadId);
  const clang::Stmt* get_next_statement_after_loop(const clang::ForStmt* for_stmt);
  const clang::Stmt* get_statement_after_compound(const clang::CompoundStmt* compound);
  const clang::Stmt* get_next_statement(const clang::Stmt* stmt);
  bool branch_contains_escape_statement(const clang::Stmt* branch);
  bool statement_causes_divergent_exit(const clang::Stmt* stmt);
  bool branch_contains_break_statement(const clang::Stmt* branch);
  bool branch_contains_continue_statement(const clang::Stmt* branch);
  bool branch_contains_return_statement(const clang::Stmt* branch);

  // Legacy compatibility
  bool couldAlias(const clang::Expr *addr1, const clang::Expr *addr2) {
    return could_alias(addr1, addr2);
  }
  bool analyzeAddressExpressions(const clang::Expr *addr1,
                                 const clang::Expr *addr2) {
    return analyze_address_expressions(addr1, addr2);
  }
  bool isDisjointLaneBasedAccess(const clang::Expr *addr1,
                                 const clang::Expr *addr2) {
    return is_disjoint_lane_based_access(addr1, addr2);
  }
  bool isConstantOffset(const clang::Expr *expr, int64_t &offset) {
    return is_constant_offset(expr, offset);
  }

  // Rust-style memory operation classification
  MemoryAccessPattern classify_memory_access(const MemoryOperation &op);
  bool is_commutative_memory_operation(const MemoryOperation &op);
  bool requires_atomicity(const MemoryOperation &op);

  // Legacy compatibility
  MemoryAccessPattern classifyMemoryAccess(const MemoryOperation &op) {
    return classify_memory_access(op);
  }
  bool isCommutativeMemoryOperation(const MemoryOperation &op) {
    return is_commutative_memory_operation(op);
  }
  bool requiresAtomicity(const MemoryOperation &op) {
    return requires_atomicity(op);
  }

  // Rust-style data flow analysis
  std::set<std::string> get_alias_set(const std::string &resource_name);
  void build_memory_dependency_graph();
  ValidationResult analyze_memory_dependencies();

  // Legacy compatibility
  std::set<std::string> getAliasSet(const std::string &resourceName) {
    return get_alias_set(resourceName);
  }
  void buildMemoryDependencyGraph() { build_memory_dependency_graph(); }
  ValidationResult analyzeMemoryDependencies() {
    return analyze_memory_dependencies();
  }

  clang::ASTContext &context_;
  std::vector<MemoryOperation> memory_operations_;
  std::map<std::string, MemoryAccessPattern> memory_access_patterns_;
  std::map<std::pair<std::string, std::string>, bool> alias_cache_;
  std::vector<clang::SourceLocation> barrier_locations_;
  std::map<int, std::vector<MemoryOperation *>> thread_operations_;
};

// Standalone wave operation validator
class WaveOperationValidator {
public:
  // Wave operation type enumeration
  enum class WaveOperationType {
    Reduction, // WaveActiveSum, WaveActiveMax, etc.
    Broadcast, // WaveReadLaneAt, etc. (forbidden)
    Query,     // WaveGetLaneIndex, WaveGetLaneCount
    Ballot,    // WaveBallot, etc. (forbidden)
    Prefix,    // WavePrefixSum, etc. (forbidden)
    Unknown
  };

  explicit WaveOperationValidator(clang::ASTContext &context)
      : context_(context) {}

  // Rust-style constructor
  static Box<WaveOperationValidator> create(clang::ASTContext &context) {
    return make_box<WaveOperationValidator>(context);
  }

  // Rust-style wave operation validation
  ValidationResult
  validate_wave_call(const clang::CallExpr *call,
                     const ControlFlowAnalyzer::ControlFlowState &cf_state);
  bool is_wave_operation(const clang::CallExpr *call);
  bool is_order_independent_wave_op(const std::string &func_name) const;
  bool requires_full_participation(const std::string &func_name);

  // Legacy compatibility
  ValidationResult
  validateWaveCall(const clang::CallExpr *call,
                   const ControlFlowAnalyzer::ControlFlowState &cfState) {
    return validate_wave_call(call, cfState);
  }
  bool isWaveOperation(const clang::CallExpr *call) {
    return is_wave_operation(call);
  }
  bool isOrderIndependentWaveOp(const std::string &funcName) {
    return is_order_independent_wave_op(funcName);
  }
  bool isOrderDependentWaveOp(const std::string &funcName) {
    return !is_order_independent_wave_op(funcName);
  }
  bool requiresFullParticipation(const std::string &funcName) {
    return requires_full_participation(funcName);
  }

private:
  // Rust-style advanced wave operation analysis
  ValidationResult analyze_wave_participation(
      const clang::CallExpr *call,
      const ControlFlowAnalyzer::ControlFlowState &cf_state);
  ValidationResult validate_wave_arguments(const clang::CallExpr *call,
                                           const std::string &func_name);
  ValidationResult check_wave_operation_context(const clang::CallExpr *call);

  // Legacy compatibility
  ValidationResult analyzeWaveParticipation(
      const clang::CallExpr *call,
      const ControlFlowAnalyzer::ControlFlowState &cfState) {
    return analyze_wave_participation(call, cfState);
  }
  ValidationResult validateWaveArguments(const clang::CallExpr *call,
                                         const std::string &funcName) {
    return validate_wave_arguments(call, funcName);
  }
  ValidationResult checkWaveOperationContext(const clang::CallExpr *call) {
    return check_wave_operation_context(call);
  }

  // Rust-style wave operation classification
  WaveOperationType classify_wave_operation(const std::string &func_name);
  bool is_reduction_operation(const std::string &func_name);
  bool is_broadcast_operation(const std::string &func_name);
  bool is_query_operation(const std::string &func_name);

  // Legacy compatibility
  WaveOperationType classifyWaveOperation(const std::string &funcName) {
    return classify_wave_operation(funcName);
  }
  bool isReductionOperation(const std::string &funcName) {
    return is_reduction_operation(funcName);
  }
  bool isBroadcastOperation(const std::string &funcName) {
    return is_broadcast_operation(funcName);
  }
  bool isQueryOperation(const std::string &funcName) {
    return is_query_operation(funcName);
  }

  // Rust-style context analysis
  bool is_in_uniform_control_flow(
      const ControlFlowAnalyzer::ControlFlowState &cf_state);
  bool is_in_divergent_control_flow(
      const ControlFlowAnalyzer::ControlFlowState &cf_state);
  int calculate_divergence_level(
      const ControlFlowAnalyzer::ControlFlowState &cf_state);

  // Legacy compatibility
  bool
  isInUniformControlFlow(const ControlFlowAnalyzer::ControlFlowState &cfState) {
    return is_in_uniform_control_flow(cfState);
  }
  bool isInDivergentControlFlow(
      const ControlFlowAnalyzer::ControlFlowState &cfState) {
    return is_in_divergent_control_flow(cfState);
  }
  int calculateDivergenceLevel(
      const ControlFlowAnalyzer::ControlFlowState &cfState) {
    return calculate_divergence_level(cfState);
  }

  clang::ASTContext &context_;
  std::map<std::string, WaveOperationType> wave_operation_types_;
};

// Main HLSL validator with full AST implementation
class MiniHLSLValidator {
public:
  MiniHLSLValidator();
  ~MiniHLSLValidator() = default;

  // Rust-style main validation methods
  ValidationResult validate_program(const Program *program);
  ValidationResult validate_function(const Function *func);
  ValidationResult validate_source(const std::string &hlsl_source);

  // Legacy compatibility
  ValidationResult validateProgram(const Program *program) {
    return validate_program(program);
  }
  ValidationResult validateFunction(const Function *func) {
    return validate_function(func);
  }
  ValidationResult validateSource(const std::string &hlslSource) {
    return validate_source(hlslSource);
  }

  // Rust-style AST-based methods
  ValidationResult validate_ast(clang::TranslationUnitDecl *tu,
                                clang::ASTContext &context);
  ValidationResult
  validate_source_with_full_ast(const std::string &hlsl_source,
                                const std::string &filename = "shader.hlsl");

  // Legacy compatibility
  ValidationResult validateAST(clang::TranslationUnitDecl *tu,
                               clang::ASTContext &context) {
    return validate_ast(tu, context);
  }
  ValidationResult
  validateSourceWithFullAST(const std::string &hlslSource,
                            const std::string &filename = "shader.hlsl") {
    return validate_source_with_full_ast(hlslSource, filename);
  }

  // Rust-style formal proof integration
  std::string generate_formal_proof_alignment(const Program *program);
  FormalProofMapping map_to_formal_proof(const ValidationResult &result,
                                         const Program *program);

  // Legacy compatibility
  std::string generateFormalProofAlignment(const Program *program) {
    return generate_formal_proof_alignment(program);
  }
  FormalProofMapping mapToFormalProof(const ValidationResult &result,
                                      const Program *program) {
    return map_to_formal_proof(result, program);
  }

  // Rust-style advanced analysis capabilities
  ValidationResult perform_full_static_analysis(clang::TranslationUnitDecl *tu,
                                                clang::ASTContext &context);
  ValidationResult analyze_order_independence(clang::TranslationUnitDecl *tu,
                                              clang::ASTContext &context);
  ValidationResult
  generate_optimization_suggestions(clang::TranslationUnitDecl *tu,
                                    clang::ASTContext &context);

  // Legacy compatibility
  ValidationResult performFullStaticAnalysis(clang::TranslationUnitDecl *tu,
                                             clang::ASTContext &context) {
    return perform_full_static_analysis(tu, context);
  }
  ValidationResult analyzeOrderIndependence(clang::TranslationUnitDecl *tu,
                                            clang::ASTContext &context) {
    return analyze_order_independence(tu, context);
  }
  ValidationResult
  generateOptimizationSuggestions(clang::TranslationUnitDecl *tu,
                                  clang::ASTContext &context) {
    return generate_optimization_suggestions(tu, context);
  }

private:
  // Rust-style analyzer instances (using Box<T> instead of std::unique_ptr)
  Box<DeterministicExpressionAnalyzer> expression_analyzer_;
  Box<ControlFlowAnalyzer> control_flow_analyzer_;
  Box<MemorySafetyAnalyzer> memory_analyzer_;
  Box<WaveOperationValidator> wave_validator_;

  // DXC-based validation methods
  ValidationResult perform_minihlsl_validation(const std::string &hlsl_source);
  ValidationResult
  validate_forbidden_wave_operations(const std::string &hlsl_source);
  ValidationResult
  validate_control_flow_patterns(const std::string &hlsl_source);
  ValidationResult
  validate_wave_operation_usage(const std::string &hlsl_source);
  ValidationResult
  validate_memory_safety_patterns(const std::string &hlsl_source);

  // Parse HLSL and perform complete DBEG validation using standard Clang pattern
  ValidationResult
  parse_and_validate_hlsl(const std::string &source,
                          const std::string &filename);

  Box<clang::CompilerInstance> setup_complete_compiler();

  // Legacy compatibility - now returns validation result instead of AST
  ValidationResult
  parseHLSLWithCompleteAST(const std::string &source,
                           const std::string &filename) {
    return parse_and_validate_hlsl(source, filename);
  }
  std::unique_ptr<clang::CompilerInstance> setupCompleteCompiler() {
    return setup_complete_compiler();
  }

  // Rust-style validation workflow
  ValidationResult run_complete_validation(clang::TranslationUnitDecl *tu,
                                           clang::ASTContext &context);
  ValidationResult validate_all_constraints(clang::TranslationUnitDecl *tu,
                                            clang::ASTContext &context);
  ValidationResult generate_comprehensive_report(const ValidationResult &result,
                                                 clang::TranslationUnitDecl *tu,
                                                 clang::ASTContext &context);

  // Legacy compatibility
  ValidationResult runCompleteValidation(clang::TranslationUnitDecl *tu,
                                         clang::ASTContext &context) {
    return run_complete_validation(tu, context);
  }
  ValidationResult validateAllConstraints(clang::TranslationUnitDecl *tu,
                                          clang::ASTContext &context) {
    return validate_all_constraints(tu, context);
  }
  ValidationResult generateComprehensiveReport(const ValidationResult &result,
                                               clang::TranslationUnitDecl *tu,
                                               clang::ASTContext &context) {
    return generate_comprehensive_report(result, tu, context);
  }

  // Rust-style analysis coordination
  void initialize_analyzers(clang::ASTContext &context);
  ValidationResult coordinate_analysis(clang::TranslationUnitDecl *tu,
                                       clang::ASTContext &context);
  ValidationResult
  consolidate_results(const std::vector<ValidationResult> &results);

  // Legacy compatibility
  void initializeAnalyzers(clang::ASTContext &context) {
    initialize_analyzers(context);
  }
  ValidationResult coordinateAnalysis(clang::TranslationUnitDecl *tu,
                                      clang::ASTContext &context) {
    return coordinate_analysis(tu, context);
  }
  ValidationResult
  consolidateResults(const std::vector<ValidationResult> &results) {
    return consolidate_results(results);
  }
};

// Rust-style factory for creating validators
class ValidatorFactory {
public:
  // Rust-style factory methods
  static Box<MiniHLSLValidator> create_validator();
  static Box<MiniHLSLValidator> create_mini_hlsl_validator();

  // Create validator with specific analyzers (Rust-style)
  static Box<MiniHLSLValidator> create_validator_with_analyzers(
      Box<DeterministicExpressionAnalyzer> expression_analyzer,
      Box<ControlFlowAnalyzer> control_flow_analyzer,
      Box<MemorySafetyAnalyzer> memory_analyzer,
      Box<WaveOperationValidator> wave_validator);

  // Legacy compatibility
  static std::unique_ptr<MiniHLSLValidator> createValidator() {
    return create_validator();
  }
  static std::unique_ptr<MiniHLSLValidator> createMiniHLSLValidator() {
    return create_mini_hlsl_validator();
  }
};

} // namespace minihlsl