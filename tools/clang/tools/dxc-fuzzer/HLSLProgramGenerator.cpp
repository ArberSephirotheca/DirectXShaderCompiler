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

std::unique_ptr<interpreter::Statement> 
IncrementalGenerator::applyMutationsSelectively(
    const interpreter::Statement* stmt,
    ProgramState& state,
    FuzzedDataProvider& provider) {
    
    // We can't apply trace-guided mutations during generation
    // This would need to be done after execution
    // For now, return nullptr to skip mutations during generation
    return nullptr;
}

void IncrementalGenerator::handleMutationBufferRequirements(
    const interpreter::Statement* stmt,
    ProgramState& state) {
    
    std::cerr << "DEBUG: handleMutationBufferRequirements called\n";
    
    // Get mutation metadata to check what type was applied
    auto mutationMeta = mutationTracker->getMetadata(stmt);
    if (!mutationMeta) {
        std::cerr << "DEBUG: No mutation metadata found\n";
        return;
    }
    std::cerr << "DEBUG: Found metadata with " << mutationMeta->appliedMutations.size() << " mutations\n";
    
    // Check if we need to handle WaveParticipantTrackingMutation buffer
    bool needsParticipantBuffer = false;
    for (const auto& mut : mutationMeta->appliedMutations) {
        std::cerr << "DEBUG: Checking mutation type: " << static_cast<int>(mut) << "\n";
        if (mut == MutationType::ParticipantTracking) {
            needsParticipantBuffer = true;
            std::cerr << "DEBUG: Found ParticipantTracking mutation\n";
            break;
        }
    }
    
    if (needsParticipantBuffer) {
        // Add buffer if it doesn't exist
        bool hasBuffer = false;
        for (const auto& buffer : state.program.globalBuffers) {
            if (buffer.name == "_participant_check_sum") {
                hasBuffer = true;
                break;
            }
        }
        
        if (!hasBuffer) {
            std::cerr << "DEBUG: Adding _participant_check_sum buffer\n";
            interpreter::GlobalBufferDecl participantBuffer;
            participantBuffer.name = "_participant_check_sum";
            participantBuffer.bufferType = "RWBuffer";
            participantBuffer.elementType = interpreter::HLSLType::Uint;
            participantBuffer.size = state.program.getTotalThreads();
            participantBuffer.registerIndex = 1;
            participantBuffer.isReadWrite = true;
            state.program.globalBuffers.push_back(participantBuffer);
            std::cerr << "DEBUG: Buffer added, total buffers: " << state.program.globalBuffers.size() << "\n";
            
            // Add initialization at the beginning
            std::cerr << "DEBUG: Adding _participant_check_sum initialization\n";
            auto tidX = std::make_unique<interpreter::DispatchThreadIdExpr>(0);
            auto zero = std::make_unique<interpreter::LiteralExpr>(0);
            state.program.statements.insert(
                state.program.statements.begin(),
                std::make_unique<interpreter::ArrayAssignStmt>(
                    "_participant_check_sum", std::move(tidX), std::move(zero))
            );
        } else {
            std::cerr << "DEBUG: _participant_check_sum buffer already exists, skipping initialization\n";
        }
        // If buffer already exists, we assume initialization was already added in the first round
        // No need to add initialization again
    }
}

void IncrementalGenerator::applyMutationsToNew(ProgramState& state, FuzzedDataProvider& provider) {
    // This method is now deprecated - mutations are applied inline during generation
    // Kept for backward compatibility
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
        
        // Don't advance round yet - we want to apply mutations to statements from this round
        
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
            
            // Register with mutation tracker BEFORE potential mutation
            mutationTracker->registerStatement(stmt.get(), meta);
            
            // Apply mutations selectively
            auto mutatedStmt = applyMutationsSelectively(stmt.get(), state, provider);
            
            if (mutatedStmt) {
                // Show the mutation transformation
                std::cerr << "\n=== Mutation Applied ===\n";
                std::cerr << "Original: " << stmt->toString() << "\n";
                std::cerr << "Mutated:  " << mutatedStmt->toString() << "\n";
                std::cerr << "=========================\n";
            }
            
            if (mutatedStmt) {
                // Copy metadata to the mutated statement, including any mutations that were recorded
                auto existingMeta = mutationTracker->getMetadata(mutatedStmt.get());
                if (existingMeta) {
                    std::cerr << "DEBUG: Found existing metadata on mutated stmt with " 
                              << existingMeta->appliedMutations.size() << " mutations\n";
                    // Copy applied mutations from the mutated statement's metadata
                    meta.appliedMutations = existingMeta->appliedMutations;
                    meta.mutationHistory = existingMeta->mutationHistory;
                    meta.mutatedWaveOpIndices = existingMeta->mutatedWaveOpIndices;
                } else {
                    std::cerr << "DEBUG: No existing metadata on mutated stmt\n";
                }
                
                // Register the updated metadata
                mutationTracker->registerStatement(mutatedStmt.get(), meta);
                
                // Handle buffer requirements for the mutation
                handleMutationBufferRequirements(mutatedStmt.get(), state);
                
                state.program.statements.push_back(std::move(mutatedStmt));
                
                // Record mutation in round info
                auto mutationMeta = mutationTracker->getMetadata(state.program.statements.back().get());
                if (mutationMeta && !mutationMeta->appliedMutations.empty()) {
                    roundInfo.appliedMutations.insert(
                        roundInfo.appliedMutations.end(),
                        mutationMeta->appliedMutations.begin(),
                        mutationMeta->appliedMutations.end()
                    );
                }
                std::cerr << "DEBUG: Applied mutation to statement in round " << round << "\n";
            } else {
                state.program.statements.push_back(std::move(stmt));
                if (!meta.waveOps.empty()) {
                    std::cerr << "DEBUG: No mutation applied to statement with " 
                              << meta.waveOps.size() << " wave ops in round " << round << "\n";
                }
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
        provider.ConsumeIntegralInRange<int>(0, 4));
    
    bool isLoop = (blockType == ControlFlowGenerator::BlockSpec::FOR_LOOP ||
                  blockType == ControlFlowGenerator::BlockSpec::WHILE_LOOP);
    
    ControlFlowGenerator::BlockSpec spec{
        blockType,
        std::move(pattern),
        isLoop && provider.ConsumeBool(),
        isLoop && provider.ConsumeBool(),
        provider.ConsumeIntegralInRange<uint32_t>(0, 2)
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