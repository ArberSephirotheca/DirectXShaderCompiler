#include "HLSLProgramGenerator.h"
#include "HLSLParticipantPatterns.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include <algorithm>

// External verbosity flag from fuzzer
extern int g_verbosity;

namespace minihlsl {
namespace fuzzer {

// Helper to find wave operations in a statement
std::vector<const interpreter::WaveActiveOp*> findWaveOpsInStatement(const interpreter::Statement* stmt) {
    std::vector<const interpreter::WaveActiveOp*> ops;
    
    // This is a simplified version - in real implementation we'd need a proper visitor
    // For now, check common patterns
    if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
        if (auto waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assignStmt->getExpression())) {
            ops.push_back(waveOp);
        }
    } else if (auto varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
        if (varDecl->getInit()) {
            if (auto waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(varDecl->getInit())) {
                ops.push_back(waveOp);
            }
        }
    }
    
    return ops;
}

// IncrementalGenerator implementation
IncrementalGenerator::IncrementalGenerator() 
    : cfGenerator(std::make_unique<ControlFlowGenerator>()),
      mutationTracker(std::make_unique<MutationTracker>()) {
}

IncrementalGenerator::~IncrementalGenerator() = default;

std::unique_ptr<ParticipantPattern> IncrementalGenerator::createPattern(FuzzedDataProvider& provider) {
    return createRandomPattern(provider);
}

void IncrementalGenerator::initializeBaseProgram(ProgramState& state, FuzzedDataProvider& provider) {
    // Set thread configuration
    state.program.numThreadsX = 4; // Or from provider
    state.program.numThreadsY = 1;
    state.program.numThreadsZ = 1;
    
    // Set wave size if specified
    if (provider.ConsumeBool()) {
        state.program.waveSize = provider.ConsumeBool() ? 32 : 64;
    }
    
    // Add SV_DispatchThreadID parameter
    interpreter::ParameterSig tidParam;
    tidParam.name = "tid";
    tidParam.type = interpreter::HLSLType::Uint3;
    tidParam.semantic = interpreter::HLSLSemantic::SV_DispatchThreadID;
    state.program.entryInputs.parameters.push_back(tidParam);
    state.program.entryInputs.hasDispatchThreadID = true;
    state.program.entryInputs.dispatchThreadIdParamName = "tid";
    
    // Add basic variable declarations
    auto resultDecl = std::make_unique<interpreter::VarDeclStmt>(
        "result", 
        interpreter::HLSLType::Uint,
        std::make_unique<interpreter::LiteralExpr>(0)
    );
    
    // Create and register metadata for the statement
    StatementMetadata meta;
    meta.originalIndex = 0;
    meta.currentIndex = 0;
    meta.isNewlyAdded = false;
    meta.generationRound = 0;
    meta.context = StatementMetadata::TopLevel;
    meta.nestingLevel = 0;
    meta.writesVariables.insert("result");
    
    // Register with mutation tracker
    auto* stmtPtr = resultDecl.get();
    state.program.statements.push_back(std::move(resultDecl));
    state.declaredVariables.insert("result");
    
    if (mutationTracker) {
        mutationTracker->registerStatement(stmtPtr, meta);
    }
}

// findWaveOps is now handled by the mutation tracker's findAllWaveOps


ProgramState IncrementalGenerator::generateIncremental(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);
    ProgramState state;
    
    // Initialize base program
    initializeBaseProgram(state, provider);
    
    // Determine number of generation rounds
    uint32_t rounds = provider.ConsumeIntegralInRange<uint32_t>(1, 5);
    
    for (uint32_t round = 0; round < rounds; ++round) {
        GenerationRound roundInfo;
        roundInfo.roundNumber = round;
        
        // Don't advance round yet - we want to apply mutations to statements from this round
        
        // Generate new control flow block
        auto pattern = createPattern(provider);
        auto blockType = static_cast<ControlFlowGenerator::BlockSpec::Type>(
            provider.ConsumeIntegralInRange<int>(0, 5));  // 6 types: IF, IF_ELSE, NESTED_IF, FOR_LOOP, WHILE_LOOP, CASCADING_IF_ELSE
        
        // Only allow break/continue in loop contexts
        bool isLoop = (blockType == ControlFlowGenerator::BlockSpec::FOR_LOOP ||
                      blockType == ControlFlowGenerator::BlockSpec::WHILE_LOOP);
        
        ControlFlowGenerator::BlockSpec spec{
            blockType,
            std::move(pattern),
            isLoop && provider.ConsumeBool(),  // includeBreak - only for loops
            isLoop && provider.ConsumeBool(),  // includeContinue - only for loops
            provider.ConsumeIntegralInRange<uint32_t>(0, 2),  // nestingDepth
            // numBranches for cascading if-else
            blockType == ControlFlowGenerator::BlockSpec::CASCADING_IF_ELSE ? 
                provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3
        };
        
        auto newStatements = cfGenerator->generateBlock(spec, state, provider);
        
        // Add to program and update metadata
        size_t insertPos = state.program.statements.size();
        
        // Check if there's a pending statement to add first
        if (state.pendingStatement) {
            // Create metadata for pending statement
            StatementMetadata meta;
            meta.originalIndex = insertPos;
            meta.currentIndex = insertPos;
            meta.isNewlyAdded = true;
            meta.generationRound = round;
            meta.waveOps = ::minihlsl::fuzzer::findAllWaveOps(state.pendingStatement.get());
            meta.context = StatementMetadata::TopLevel;
            meta.nestingLevel = 0;
            
            // Register with mutation tracker
            mutationTracker->registerStatement(state.pendingStatement.get(), meta);
            
            roundInfo.addedStatementIndices.push_back(insertPos);
            state.program.statements.push_back(std::move(state.pendingStatement));
            insertPos++;
        }
        
        // Process new statements
        for (auto& stmt : newStatements) {
            // Create metadata for new statement
            StatementMetadata meta;
            meta.originalIndex = insertPos;
            meta.currentIndex = insertPos;
            meta.isNewlyAdded = true;
            meta.generationRound = round;
            meta.waveOps = ::minihlsl::fuzzer::findAllWaveOps(stmt.get());
            meta.context = StatementMetadata::TopLevel;  // TODO: Set proper context
            meta.nestingLevel = 0;  // TODO: Calculate proper nesting
            
            // Register with mutation tracker
            mutationTracker->registerStatement(stmt.get(), meta);
            
            // Add the statement to the program
            state.program.statements.push_back(std::move(stmt));
            if (!meta.waveOps.empty()) {
                std::cerr << "DEBUG: No mutation applied to statement with " 
                          << meta.waveOps.size() << " wave ops in round " << round << "\n";
            }
            
            roundInfo.addedStatementIndices.push_back(insertPos);
            insertPos++;
        }
        
        // Record round
        roundInfo.description = "Round " + std::to_string(round);
        state.history.push_back(roundInfo);
        
        // Now advance to next round for future mutations
        mutationTracker->advanceRound();
    }
    
    return state;
}

// ControlFlowGenerator implementation
std::vector<std::unique_ptr<interpreter::Statement>>
ControlFlowGenerator::generateBlock(const BlockSpec& spec, ProgramState& state, 
                                   FuzzedDataProvider& provider) {
    std::vector<std::unique_ptr<interpreter::Statement>> statements;
    
    switch (spec.type) {
        case BlockSpec::IF:
        case BlockSpec::IF_ELSE:
            statements.push_back(generateIf(spec, state, provider));
            break;
        case BlockSpec::FOR_LOOP:
            statements.push_back(generateForLoop(spec, state, provider));
            break;
        case BlockSpec::WHILE_LOOP:
            statements.push_back(generateWhileLoop(spec, state, provider));
            break;
        case BlockSpec::NESTED_IF:
            // Generate nested if by creating two if statements
            statements.push_back(generateIf(spec, state, provider));
            break;
        case BlockSpec::CASCADING_IF_ELSE:
            statements.push_back(generateCascadingIfElse(spec, state, provider));
            break;
    }
    
    return statements;
}

std::unique_ptr<interpreter::Statement> 
ControlFlowGenerator::generateIf(const BlockSpec& spec, ProgramState& state, 
                                FuzzedDataProvider& provider) {
    // Generate condition using pattern
    uint32_t waveSize = state.program.waveSize > 0 ? 
                       state.program.waveSize : 32;
    auto condition = spec.pattern->generateCondition(waveSize, provider);
    
    // Generate body with wave operation
    std::vector<std::unique_ptr<interpreter::Statement>> thenBody;
    
    // Always include at least one wave operation
    auto waveOp = generateWaveOperation(state, provider);
    
    // Use assignment to existing variable (result) to avoid scope issues
    // Or use a variable that was declared at the function scope
    if (!state.declaredVariables.empty() && provider.ConsumeBool()) {
        // Assign to existing variable
        std::vector<std::string> vars(state.declaredVariables.begin(), 
                                     state.declaredVariables.end());
        size_t idx = provider.ConsumeIntegralInRange<size_t>(0, vars.size() - 1);
        thenBody.push_back(std::make_unique<interpreter::AssignStmt>(
            vars[idx],
            std::move(waveOp)
        ));
    } else {
        // Always assign to 'result' which is declared at the beginning
        thenBody.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(waveOp)
        ));
    }
    
    // Break/continue should not be added in if statements - they're only valid in loops
    
    // Generate else body if requested
    std::vector<std::unique_ptr<interpreter::Statement>> elseBody;
    if (spec.type == BlockSpec::IF_ELSE) {
        // Simple assignment in else
        if (!state.declaredVariables.empty()) {
            std::vector<std::string> vars(state.declaredVariables.begin(), 
                                         state.declaredVariables.end());
            size_t idx = provider.ConsumeIntegralInRange<size_t>(0, vars.size() - 1);
            elseBody.push_back(std::make_unique<interpreter::AssignStmt>(
                vars[idx],
                std::make_unique<interpreter::LiteralExpr>(0)
            ));
        } else {
            elseBody.push_back(std::make_unique<interpreter::AssignStmt>(
                "result",
                std::make_unique<interpreter::LiteralExpr>(0)
            ));
        }
    }
    
    return std::make_unique<interpreter::IfStmt>(
        std::move(condition), 
        std::move(thenBody),
        std::move(elseBody)
    );
}

std::unique_ptr<interpreter::Statement>
ControlFlowGenerator::generateForLoop(const BlockSpec& spec, ProgramState& state,
                                     FuzzedDataProvider& provider) {
    // Initialize loop variable
    std::string loopVar = "i" + std::to_string(state.nextVarIndex++);
    state.declaredVariables.insert(loopVar);
    
    // Initialize expression for the loop variable
    auto initExpr = std::make_unique<interpreter::LiteralExpr>(0);
    
    // Loop condition - limit to 3 iterations to avoid false deadlock detection
    uint32_t loopCount = provider.ConsumeIntegralInRange<uint32_t>(2, 3);
    auto condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(loopVar),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(loopCount)),
        interpreter::BinaryOpExpr::Lt
    );
    
    // Increment expression - must be an assignment to update the loop variable
    auto increment = std::make_unique<interpreter::AssignExpr>(
        loopVar,
        std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(loopVar),
            std::make_unique<interpreter::LiteralExpr>(1),
            interpreter::BinaryOpExpr::Add
        )
    );
    
    // Generate body with participant pattern
    std::vector<std::unique_ptr<interpreter::Statement>> body;
    
    // Use pattern for conditional wave operation
    uint32_t waveSize = state.program.waveSize > 0 ? 
                       state.program.waveSize : 32;
    auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
    auto waveOp = generateWaveOperation(state, provider);
    // Use assignment to existing variable to avoid scope issues
    waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
        "result",
        std::move(waveOp)
    ));
    
    // Add optional continue
    if (spec.includeContinue && provider.ConsumeBool()) {
        auto continueCondition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(loopVar),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(loopCount / 2)),
            interpreter::BinaryOpExpr::Eq
        );
        
        std::vector<std::unique_ptr<interpreter::Statement>> continueBody;
        continueBody.push_back(std::make_unique<interpreter::ContinueStmt>());
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(continueCondition),
            std::move(continueBody)
        ));
    }
    
    body.push_back(std::make_unique<interpreter::IfStmt>(
        std::move(participantCondition),
        std::move(waveBody)
    ));
    
    // Add optional break
    if (spec.includeBreak && provider.ConsumeBool()) {
        auto breakCondition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(loopVar),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(loopCount - 1)),
            interpreter::BinaryOpExpr::Eq
        );
        
        std::vector<std::unique_ptr<interpreter::Statement>> breakBody;
        breakBody.push_back(std::make_unique<interpreter::BreakStmt>());
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(breakCondition),
            std::move(breakBody)
        ));
    }
    
    return std::make_unique<interpreter::ForStmt>(
        loopVar,
        std::move(initExpr),
        std::move(condition),
        std::move(increment),
        std::move(body)
    );
}

std::unique_ptr<interpreter::Statement>
ControlFlowGenerator::generateWhileLoop(const BlockSpec& spec, ProgramState& state,
                                       FuzzedDataProvider& provider) {
    // Create loop counter variable
    std::string counterVar = "counter" + std::to_string(state.nextVarIndex++);
    
    // Add counter initialization before the loop
    auto counterInit = std::make_unique<interpreter::VarDeclStmt>(
        counterVar,
        interpreter::HLSLType::Uint,
        std::make_unique<interpreter::LiteralExpr>(0)
    );
    state.declaredVariables.insert(counterVar);
    
    // While condition - limit to 3 iterations to avoid false deadlock detection
    uint32_t maxIterations = provider.ConsumeIntegralInRange<uint32_t>(2, 3);
    auto condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(counterVar),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(maxIterations)),
        interpreter::BinaryOpExpr::Lt
    );
    
    // Body
    std::vector<std::unique_ptr<interpreter::Statement>> body;
    
    // Increment counter
    body.push_back(std::make_unique<interpreter::AssignStmt>(
        counterVar,
        std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(counterVar),
            std::make_unique<interpreter::LiteralExpr>(1),
            interpreter::BinaryOpExpr::Add
        )
    ));
    
    // Add wave operation with pattern
    uint32_t waveSize = state.program.waveSize > 0 ? 
                       state.program.waveSize : 32;
    auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
    auto waveOp = generateWaveOperation(state, provider);
    // Use assignment to existing variable to avoid scope issues
    waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
        "result",
        std::move(waveOp)
    ));
    
    body.push_back(std::make_unique<interpreter::IfStmt>(
        std::move(participantCondition),
        std::move(waveBody)
    ));
    
    // Create a compound statement to hold both the init and the while
    std::vector<std::unique_ptr<interpreter::Statement>> statements;
    statements.push_back(std::move(counterInit));
    statements.push_back(std::make_unique<interpreter::WhileStmt>(
        std::move(condition),
        std::move(body)
    ));
    
    // Store the counter init to be added before the while loop
    state.pendingStatement = std::move(statements[0]);
    
    // Return just the while statement
    return std::move(statements[1]);
}

std::unique_ptr<interpreter::Statement>
ControlFlowGenerator::generateCascadingIfElse(const BlockSpec& spec, ProgramState& state,
                                            FuzzedDataProvider& provider) {
    // Get the number of branches (default to 3 if not specified)
    uint32_t numBranches = spec.numBranches > 0 ? spec.numBranches : 3;
    uint32_t waveSize = state.program.waveSize > 0 ? 
                       state.program.waveSize : 32;
    
    // Generate unique participant patterns for each branch
    std::vector<std::unique_ptr<ParticipantPattern>> patterns;
    std::vector<std::unique_ptr<interpreter::Expression>> conditions;
    
    // Generate different patterns for each branch
    for (uint32_t i = 0; i < numBranches; ++i) {
        auto pattern = createRandomPattern(provider);
        auto condition = pattern->generateCondition(waveSize, provider);
        patterns.push_back(std::move(pattern));
        conditions.push_back(std::move(condition));
    }
    
    // Build the cascading if-else-if structure from the bottom up
    std::unique_ptr<interpreter::IfStmt> result = nullptr;
    
    for (int i = numBranches - 1; i >= 0; --i) {
        // Generate body with a different wave operation for each branch
        std::vector<std::unique_ptr<interpreter::Statement>> body;
        
        // Choose a different wave operation type for each branch
        interpreter::WaveActiveOp::OpType opType;
        switch (i % 4) {
            case 0: opType = interpreter::WaveActiveOp::Sum; break;
            case 1: opType = interpreter::WaveActiveOp::Product; break;
            case 2: opType = interpreter::WaveActiveOp::Min; break;
            case 3: opType = interpreter::WaveActiveOp::Max; break;
        }
        
        // Create wave operation with different input for each branch
        std::unique_ptr<interpreter::Expression> input;
        if (provider.ConsumeBool()) {
            // Use lane index modified by branch number
            input = std::make_unique<interpreter::BinaryOpExpr>(
                std::make_unique<interpreter::LaneIndexExpr>(),
                std::make_unique<interpreter::LiteralExpr>(i + 1),
                interpreter::BinaryOpExpr::Add
            );
        } else {
            // Use different literal for each branch
            input = std::make_unique<interpreter::LiteralExpr>(i + 1);
        }
        
        auto waveOp = std::make_unique<interpreter::WaveActiveOp>(std::move(input), opType);
        
        // Assign to result variable
        body.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(waveOp)
        ));
        
        // Create else body (which contains the previous if-else chain)
        std::vector<std::unique_ptr<interpreter::Statement>> elseBody;
        if (result != nullptr) {
            elseBody.push_back(std::move(result));
        }
        
        // Create new if statement
        result = std::make_unique<interpreter::IfStmt>(
            std::move(conditions[i]),
            std::move(body),
            std::move(elseBody)
        );
    }
    
    return result;
}

std::unique_ptr<interpreter::Expression>
ControlFlowGenerator::generateWaveOperation(ProgramState& state, FuzzedDataProvider& provider) {
    // Choose operation type
    interpreter::WaveActiveOp::OpType opType;
    uint32_t opChoice = provider.ConsumeIntegralInRange<uint32_t>(0, 3);
    switch (opChoice) {
        case 0: opType = interpreter::WaveActiveOp::Sum; break;
        case 1: opType = interpreter::WaveActiveOp::Product; break;
        case 2: opType = interpreter::WaveActiveOp::Min; break;
        case 3: opType = interpreter::WaveActiveOp::Max; break;
        default: opType = interpreter::WaveActiveOp::Sum; break;
    }
    
    // Choose input - either variable or expression
    std::unique_ptr<interpreter::Expression> input;
    if (!state.declaredVariables.empty() && provider.ConsumeBool()) {
        // Use existing variable, but filter out loop variables that may be out of scope
        std::vector<std::string> validVars;
        for (const auto& var : state.declaredVariables) {
            // Skip loop counter variables (they start with 'i' or 'counter')
            if (var.substr(0, 1) != "i" && var.substr(0, 7) != "counter") {
                validVars.push_back(var);
            }
        }
        
        if (!validVars.empty()) {
            size_t idx = provider.ConsumeIntegralInRange<size_t>(0, validVars.size() - 1);
            input = std::make_unique<interpreter::VariableExpr>(validVars[idx]);
        } else {
            // Fallback to literal if no valid variables
            input = std::make_unique<interpreter::LiteralExpr>(
                provider.ConsumeIntegralInRange<int>(1, 10));
        }
    } else {
        // Use tid.x or literal
        if (provider.ConsumeBool()) {
            // Use DispatchThreadIdExpr for tid.x (component 0)
            input = std::make_unique<interpreter::DispatchThreadIdExpr>(0);
        } else {
            input = std::make_unique<interpreter::LiteralExpr>(
                provider.ConsumeIntegralInRange<int>(1, 10));
        }
    }
    
    return std::make_unique<interpreter::WaveActiveOp>(std::move(input), opType);
}

// Utility functions - these would typically be in MiniHLSLInterpreterFuzzer.cpp
// but are defined here for completeness

interpreter::Program prepareProgramForExecution(interpreter::Program program) {
    // For now, just return the program as-is
    // In a full implementation, this would prepare the program for execution
    // by adding any necessary initialization
    return program;
}

void saveBugReport(const uint8_t* data, size_t size, 
                   const interpreter::Program& program,
                   const interpreter::MiniHLSLInterpreter::VerificationResult& verification) {
    // Implementation would save the bug report to disk
    // For now, just print to console
    std::cerr << "Bug found! Program is order-dependent.\n";
    std::cerr << "Program:\n" << serializeProgramToString(program) << "\n";
}

void IncrementalGenerator::addStatementsToProgram(ProgramState& state, const uint8_t* data, size_t size, size_t offset) {
    FuzzedDataProvider provider(data + offset, size - offset);
    
    // Add one round of new statements to the existing program
    GenerationRound roundInfo;
    roundInfo.roundNumber = state.history.size();
    
    // Generate new control flow block
    auto pattern = createPattern(provider);
    auto blockType = static_cast<ControlFlowGenerator::BlockSpec::Type>(
        provider.ConsumeIntegralInRange<int>(0, 5));
    
    bool isLoop = (blockType == ControlFlowGenerator::BlockSpec::FOR_LOOP ||
                  blockType == ControlFlowGenerator::BlockSpec::WHILE_LOOP);
    
    ControlFlowGenerator::BlockSpec spec{
        blockType,
        std::move(pattern),
        isLoop && provider.ConsumeBool(),
        isLoop && provider.ConsumeBool(),
        provider.ConsumeIntegralInRange<uint32_t>(0, 2),
        // numBranches for cascading if-else
        blockType == ControlFlowGenerator::BlockSpec::CASCADING_IF_ELSE ? 
            provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3
    };
    
    auto newStatements = cfGenerator->generateBlock(spec, state, provider);
    
    // Add to program
    size_t insertPos = state.program.statements.size();
    
    for (auto& stmt : newStatements) {
        // Create metadata
        StatementMetadata meta;
        meta.originalIndex = insertPos;
        meta.currentIndex = insertPos;
        meta.isNewlyAdded = true;
        meta.generationRound = roundInfo.roundNumber;
        meta.waveOps = ::minihlsl::fuzzer::findAllWaveOps(stmt.get());
        meta.context = StatementMetadata::TopLevel;
        meta.nestingLevel = 0;
        
        mutationTracker->registerStatement(stmt.get(), meta);
        
        state.program.statements.push_back(std::move(stmt));
        roundInfo.addedStatementIndices.push_back(insertPos);
        insertPos++;
    }
    
    // Record round
    roundInfo.description = "Incremental round " + std::to_string(roundInfo.roundNumber);
    state.history.push_back(roundInfo);
    
    mutationTracker->advanceRound();
}

} // namespace fuzzer
} // namespace minihlsl