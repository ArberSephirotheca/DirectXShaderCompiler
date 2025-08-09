#include "HLSLProgramGenerator.h"
#include "HLSLParticipantPatterns.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include <algorithm>

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
    : cfGenerator(std::make_unique<ControlFlowGenerator>()) {
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
        state.program.waveSizePreferred = provider.ConsumeBool() ? 32 : 64;
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
    
    StatementMetadata meta{
        0,  // originalIndex
        0,  // currentIndex
        false,  // isNewlyAdded
        false,  // hasMutation
        MutationType::None,
        {}  // waveOps
    };
    
    state.metadata.push_back(meta);
    state.program.statements.push_back(std::move(resultDecl));
    state.declaredVariables.insert("result");
}

std::vector<const interpreter::WaveActiveOp*> IncrementalGenerator::findWaveOps(const interpreter::Statement* stmt) {
    return findWaveOpsInStatement(stmt);
}

void IncrementalGenerator::applyMutationsToNew(ProgramState& state, FuzzedDataProvider& provider) {
    // Prepare the program for mutations
    state.program = prepareProgramForExecution(std::move(state.program));
    
    // For now, skip mutations as they require ExecutionTrace
    // In a full implementation, we would:
    // 1. Execute the program to get a trace
    // 2. Apply mutations based on the trace
    // 3. Update the program with the mutations
    
    // Mark that we attempted mutations
    for (auto& meta : state.metadata) {
        if (meta.isNewlyAdded && !meta.waveOps.empty()) {
            meta.hasMutation = true;
            meta.mutationType = provider.ConsumeBool() ? 
                MutationType::LanePermutation : MutationType::ParticipantTracking;
        }
    }
}

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
        
        // Mark existing statements as not new
        for (auto& meta : state.metadata) {
            meta.isNewlyAdded = false;
        }
        
        // Generate new control flow block
        auto pattern = createPattern(provider);
        auto blockType = static_cast<ControlFlowGenerator::BlockSpec::Type>(
            provider.ConsumeIntegralInRange<int>(0, 4));  // 5 types: IF, IF_ELSE, NESTED_IF, FOR_LOOP, WHILE_LOOP
        
        // Only allow break/continue in loop contexts
        bool isLoop = (blockType == ControlFlowGenerator::BlockSpec::FOR_LOOP ||
                      blockType == ControlFlowGenerator::BlockSpec::WHILE_LOOP);
        
        ControlFlowGenerator::BlockSpec spec{
            blockType,
            std::move(pattern),
            isLoop && provider.ConsumeBool(),  // includeBreak - only for loops
            isLoop && provider.ConsumeBool(),  // includeContinue - only for loops
            provider.ConsumeIntegralInRange<uint32_t>(0, 2)  // nestingDepth
        };
        
        auto newStatements = cfGenerator->generateBlock(spec, state, provider);
        
        // Add to program and update metadata
        size_t insertPos = state.program.statements.size();
        
        // Check if there's a pending statement to add first
        if (state.pendingStatement) {
            StatementMetadata meta{
                insertPos,  // originalIndex
                insertPos,  // currentIndex
                true,       // isNewlyAdded
                false,      // hasMutation
                MutationType::None,
                {}  // no wave ops in counter init
            };
            state.metadata.push_back(meta);
            roundInfo.addedStatementIndices.push_back(insertPos);
            state.program.statements.push_back(std::move(state.pendingStatement));
            insertPos++;
        }
        for (auto& stmt : newStatements) {
            StatementMetadata meta{
                insertPos,  // originalIndex
                insertPos,  // currentIndex
                true,       // isNewlyAdded
                false,      // hasMutation
                MutationType::None,
                findWaveOps(stmt.get())
            };
            state.metadata.push_back(meta);
            roundInfo.addedStatementIndices.push_back(insertPos);
            state.program.statements.push_back(std::move(stmt));
            insertPos++;
        }
        
        // Apply mutations to new wave operations only
        applyMutationsToNew(state, provider);
        
        // Record round
        roundInfo.description = "Round " + std::to_string(round);
        state.history.push_back(roundInfo);
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
    }
    
    return statements;
}

std::unique_ptr<interpreter::Statement> 
ControlFlowGenerator::generateIf(const BlockSpec& spec, ProgramState& state, 
                                FuzzedDataProvider& provider) {
    // Generate condition using pattern
    uint32_t waveSize = state.program.waveSizePreferred > 0 ? 
                       state.program.waveSizePreferred : 32;
    auto condition = spec.pattern->generateCondition(waveSize, provider);
    
    // Generate body with wave operation
    std::vector<std::unique_ptr<interpreter::Statement>> thenBody;
    
    // Always include at least one wave operation
    auto waveOp = generateWaveOperation(state, provider);
    std::string resultVar = state.getNewVariable();
    thenBody.push_back(std::make_unique<interpreter::VarDeclStmt>(
        resultVar,
        interpreter::HLSLType::Uint,
        std::move(waveOp)
    ));
    
    // Break/continue should not be added in if statements - they're only valid in loops
    
    // Generate else body if requested
    std::vector<std::unique_ptr<interpreter::Statement>> elseBody;
    if (spec.type == BlockSpec::IF_ELSE) {
        // Simple assignment in else
        std::string elseVar = state.getNewVariable();
        elseBody.push_back(std::make_unique<interpreter::VarDeclStmt>(
            elseVar,
            interpreter::HLSLType::Uint,
            std::make_unique<interpreter::LiteralExpr>(0)
        ));
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
    
    // Loop condition
    uint32_t loopCount = provider.ConsumeIntegralInRange<uint32_t>(2, 8);
    auto condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(loopVar),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(loopCount)),
        interpreter::BinaryOpExpr::Lt
    );
    
    // Increment expression
    auto increment = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(loopVar),
        std::make_unique<interpreter::LiteralExpr>(1),
        interpreter::BinaryOpExpr::Add
    );
    
    // Generate body with participant pattern
    std::vector<std::unique_ptr<interpreter::Statement>> body;
    
    // Use pattern for conditional wave operation
    uint32_t waveSize = state.program.waveSizePreferred > 0 ? 
                       state.program.waveSizePreferred : 32;
    auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
    auto waveOp = generateWaveOperation(state, provider);
    std::string waveResult = state.getNewVariable();
    waveBody.push_back(std::make_unique<interpreter::VarDeclStmt>(
        waveResult,
        interpreter::HLSLType::Uint,
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
    
    // While condition
    uint32_t maxIterations = provider.ConsumeIntegralInRange<uint32_t>(2, 5);
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
    uint32_t waveSize = state.program.waveSizePreferred > 0 ? 
                       state.program.waveSizePreferred : 32;
    auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
    
    std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
    auto waveOp = generateWaveOperation(state, provider);
    std::string waveResult = state.getNewVariable();
    waveBody.push_back(std::make_unique<interpreter::VarDeclStmt>(
        waveResult,
        interpreter::HLSLType::Uint,
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
        // Use existing variable
        std::vector<std::string> vars(state.declaredVariables.begin(), 
                                     state.declaredVariables.end());
        size_t idx = provider.ConsumeIntegralInRange<size_t>(0, vars.size() - 1);
        input = std::make_unique<interpreter::VariableExpr>(vars[idx]);
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

} // namespace fuzzer
} // namespace minihlsl