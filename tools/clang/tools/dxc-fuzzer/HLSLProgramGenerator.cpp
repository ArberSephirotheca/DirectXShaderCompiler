#include "HLSLProgramGenerator.h"
#include "HLSLParticipantPatterns.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include "FuzzerDebug.h"
#include <algorithm>
#include <cstdlib>

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
    // Check environment variable for thread count
    const char* numThreadsEnv = std::getenv("FUZZ_NUM_THREADS");
    uint32_t totalThreads = 64; // default
    if (numThreadsEnv) {
        try {
            totalThreads = std::stoul(numThreadsEnv);
            // Clamp to reasonable values
            if (totalThreads < 1) totalThreads = 1;
            if (totalThreads > 1024) totalThreads = 1024;
        } catch (...) {
            totalThreads = 64; // fallback to default
        }
    }
    
    state.program.numThreadsX = totalThreads;
    state.program.numThreadsY = 1;
    state.program.numThreadsZ = 1;
    
    // Set wave size from environment variable if available
    const char* waveSizeEnv = std::getenv("FUZZ_WAVE_SIZE");
    if (waveSizeEnv) {
        try {
            uint32_t waveSize = std::stoul(waveSizeEnv);
            // Validate wave size
            if (waveSize == 4 || waveSize == 8 || waveSize == 16 || waveSize == 32 || waveSize == 64) {
                state.program.waveSize = waveSize;
            } else {
                state.program.waveSize = 32; // default
            }
        } catch (...) {
            state.program.waveSize = 32; // default
        }
    } else {
        // Default wave size
        state.program.waveSize = 32;
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
    // Reduced from 1-5 to 1-3 to limit program complexity
    uint32_t rounds = provider.ConsumeIntegralInRange<uint32_t>(1, 3);
    
    for (uint32_t round = 0; round < rounds; ++round) {
        GenerationRound roundInfo;
        roundInfo.roundNumber = round;
        
        // Don't advance round yet - we want to apply mutations to statements from this round
        
        // Generate new control flow block
        auto pattern = createPattern(provider);
        auto blockType = static_cast<ControlFlowGenerator::BlockSpec::Type>(
            provider.ConsumeIntegralInRange<int>(0, 9) < 4 ? provider.ConsumeIntegralInRange<int>(0, 5) : 6);  // 40% chance for SWITCH_STMT
        
        // Debug: Print selected block type
        const char* blockTypeNames[] = {"IF", "IF_ELSE", "NESTED_IF", "FOR_LOOP", "WHILE_LOOP", "CASCADING_IF_ELSE", "SWITCH_STMT"};
        FUZZER_DEBUG_LOG("Round " << round << ": Selected block type: " << blockTypeNames[blockType] << " (" << blockType << ")\n");
        
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
                provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3,
            std::nullopt,  // switchConfig - will be set below if needed
            {} // nestingContext with default initialization
        };
        
        // Configure switch-specific settings if it's a switch statement
        if (blockType == ControlFlowGenerator::BlockSpec::SWITCH_STMT) {
            ControlFlowGenerator::BlockSpec::SwitchConfig switchConfig;
            
            // Randomly select selector type
            uint32_t selectorChoice = provider.ConsumeIntegralInRange<uint32_t>(0, 2);
            switchConfig.selectorType = static_cast<ControlFlowGenerator::BlockSpec::SwitchConfig::SelectorType>(selectorChoice);
            
            // Number of cases (2-4)
            switchConfig.numCases = provider.ConsumeIntegralInRange<uint32_t>(2, 4);
            
            // 30% chance to include default case
            switchConfig.includeDefault = provider.ConsumeIntegralInRange<uint32_t>(0, 9) < 3;
            
            // For phase 1, always use break statements
            switchConfig.allCasesBreak = true;
            
            spec.switchConfig = switchConfig;
        }
        
        // Enable nesting for this block - always allow nesting to maximize complexity
        spec.nestingContext.allowNesting = true;  // Always allow nesting
        spec.nestingContext.maxDepth = provider.ConsumeIntegralInRange<uint32_t>(2, 3);  // Reduced max depth to avoid timeout
        spec.nestingContext.currentDepth = 0;
        
        auto newStatements = cfGenerator->generateBlock(spec, state, provider);
        
        // Add to program and update metadata
        size_t insertPos = state.program.statements.size();
        
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
        case BlockSpec::WHILE_LOOP: {
            auto whileStatements = generateWhileLoop(spec, state, provider);
            for (auto& stmt : whileStatements) {
                statements.push_back(std::move(stmt));
            }
            break;
        }
        case BlockSpec::NESTED_IF:
            // Generate nested if by creating two if statements
            statements.push_back(generateIf(spec, state, provider));
            break;
        case BlockSpec::CASCADING_IF_ELSE:
            statements.push_back(generateCascadingIfElse(spec, state, provider));
            break;
        case BlockSpec::SWITCH_STMT:
            statements.push_back(generateSwitch(spec, state, provider));
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
    
    // Generate then body with potential nesting
    auto thenBody = generateNestedBody(spec, state, provider);
    
    // Add break/continue if specified AND we're in a loop context
    if (spec.includeBreak && provider.ConsumeBool()) {
        // Skip adding break here - we'll add uniform breaks in the loop generation functions
        // This is because we need access to the loop variable for uniform conditions
    }
    
    // Generate else body if requested
    std::vector<std::unique_ptr<interpreter::Statement>> elseBody;
    if (spec.type == BlockSpec::IF_ELSE) {
        // Create a separate spec for else branch
        BlockSpec elseSpec{
            spec.type,
            createRandomPattern(provider),  // Create new pattern for else branch
            spec.includeBreak,
            spec.includeContinue,
            spec.nestingDepth,
            spec.numBranches,
            std::nullopt,  // switchConfig - not needed for IF_ELSE
            spec.nestingContext  // Copy nesting context
        };
        elseSpec.nestingContext.parentTypes.push_back(BlockSpec::IF_ELSE);
        
        // Generate else body with potential nesting
        elseBody = generateNestedBody(elseSpec, state, provider);
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
    // Initialize loop variable using context-aware naming
    std::string loopVar = generateLoopVariable(spec.nestingContext, BlockSpec::FOR_LOOP, state);
    
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
    
    // Generate body with potential nesting
    auto body = generateNestedBody(spec, state, provider);
    
    // Add optional continue (only if we're generating a loop)
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

std::vector<std::unique_ptr<interpreter::Statement>>
ControlFlowGenerator::generateWhileLoop(const BlockSpec& spec, ProgramState& state,
                                       FuzzedDataProvider& provider) {
    // Create loop counter variable using context-aware naming
    std::string counterVar = generateLoopVariable(spec.nestingContext, BlockSpec::WHILE_LOOP, state);
    
    // Add counter initialization before the loop
    auto counterInit = std::make_unique<interpreter::VarDeclStmt>(
        counterVar,
        interpreter::HLSLType::Uint,
        std::make_unique<interpreter::LiteralExpr>(0)
    );
    
    // While condition - limit to 3 iterations to avoid false deadlock detection
    uint32_t maxIterations = provider.ConsumeIntegralInRange<uint32_t>(2, 3);
    auto condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::VariableExpr>(counterVar),
        std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(maxIterations)),
        interpreter::BinaryOpExpr::Lt
    );
    
    // Body
    std::vector<std::unique_ptr<interpreter::Statement>> body;
    
    // Increment counter first
    body.push_back(std::make_unique<interpreter::AssignStmt>(
        counterVar,
        std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(counterVar),
            std::make_unique<interpreter::LiteralExpr>(1),
            interpreter::BinaryOpExpr::Add
        )
    ));
    
    // Generate body with potential nesting
    auto nestedStatements = generateNestedBody(spec, state, provider);
    for (auto& stmt : nestedStatements) {
        body.push_back(std::move(stmt));
    }
    
    // Add optional break/continue if we're in a loop context
    if (spec.includeBreak && provider.ConsumeBool()) {
        auto breakCondition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(counterVar),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int32_t>(maxIterations - 1)),
            interpreter::BinaryOpExpr::Eq
        );
        
        std::vector<std::unique_ptr<interpreter::Statement>> breakBody;
        breakBody.push_back(std::make_unique<interpreter::BreakStmt>());
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(breakCondition),
            std::move(breakBody)
        ));
    }
    
    // Return both the counter init and the while statement
    std::vector<std::unique_ptr<interpreter::Statement>> statements;
    statements.push_back(std::move(counterInit));
    statements.push_back(std::make_unique<interpreter::WhileStmt>(
        std::move(condition),
        std::move(body)
    ));
    
    return statements;
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
        
        // Choose a different wave operation type for each branch (excluding Product)
        interpreter::WaveActiveOp::OpType opType;
        switch (i % 3) {
            case 0: opType = interpreter::WaveActiveOp::Sum; break;
            case 1: opType = interpreter::WaveActiveOp::Min; break;
            case 2: opType = interpreter::WaveActiveOp::Max; break;
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
        
        // Use += for accumulation to see cumulative effects
        auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
        auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(currentResult),
            std::move(waveOp),
            interpreter::BinaryOpExpr::Add
        );
        body.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(addExpr)
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

// Helper to check if we're inside any loop
static bool hasEnclosingLoop(const ControlFlowGenerator::BlockSpec::NestingContext& context) {
    for (const auto& parentType : context.parentTypes) {
        if (parentType == ControlFlowGenerator::BlockSpec::FOR_LOOP || 
            parentType == ControlFlowGenerator::BlockSpec::WHILE_LOOP) {
            return true;
        }
    }
    return false;
}

static bool hasEnclosingSwitch(const ControlFlowGenerator::BlockSpec::NestingContext& context) {
    for (const auto& parentType : context.parentTypes) {
        if (parentType == ControlFlowGenerator::BlockSpec::SWITCH_STMT) {
            return true;
        }
    }
    return false;
}

// Nesting support methods
ControlFlowGenerator::BlockSpec 
ControlFlowGenerator::createNestedBlockSpec(const BlockSpec& parentSpec,
                                           ProgramState& state,
                                           FuzzedDataProvider& provider) {
    BlockSpec nestedSpec;
    
    // Copy and update nesting context
    nestedSpec.nestingContext = parentSpec.nestingContext;
    nestedSpec.nestingContext.currentDepth++;
    nestedSpec.nestingContext.parentTypes.push_back(parentSpec.type);
    
    // Don't allow nesting if we've reached max depth
    if (nestedSpec.nestingContext.currentDepth >= nestedSpec.nestingContext.maxDepth) {
        nestedSpec.nestingContext.allowNesting = false;
    } else {
        nestedSpec.nestingContext.allowNesting = true;
    }
    
    // Choose what type of control flow to nest
    if (parentSpec.type == BlockSpec::FOR_LOOP || parentSpec.type == BlockSpec::WHILE_LOOP) {
        // Check if we already have two consecutive loops in the parent chain
        // This prevents triple-nested loops (loop->loop->loop) which create excessive blocks
        // while still allowing patterns like switch->loop->loop or loop->if->loop
        bool hasConsecutiveLoops = false;
        if (nestedSpec.nestingContext.parentTypes.size() >= 2) {
            size_t n = nestedSpec.nestingContext.parentTypes.size();
            // Check if the last two parents are both loops
            auto parent1 = nestedSpec.nestingContext.parentTypes[n-1];
            auto parent2 = nestedSpec.nestingContext.parentTypes[n-2];
            if ((parent1 == BlockSpec::FOR_LOOP || parent1 == BlockSpec::WHILE_LOOP) &&
                (parent2 == BlockSpec::FOR_LOOP || parent2 == BlockSpec::WHILE_LOOP)) {
                hasConsecutiveLoops = true;
            }
        }
        
        // If we already have two consecutive loops, don't allow a third loop
        if (hasConsecutiveLoops) {
            // Only allow non-loop structures
            int choice = provider.ConsumeIntegralInRange<int>(0, 2);
            switch (choice) {
                case 0:
                    nestedSpec.type = BlockSpec::IF;
                    break;
                case 1:
                    nestedSpec.type = BlockSpec::IF_ELSE;
                    break;
                case 2:
                    nestedSpec.type = BlockSpec::SWITCH_STMT;
                    // Setup switch configuration
                    {
                        BlockSpec::SwitchConfig switchConfig;
                        switchConfig.selectorType = static_cast<BlockSpec::SwitchConfig::SelectorType>(
                            provider.ConsumeIntegralInRange<int>(0, 2));
                        switchConfig.numCases = provider.ConsumeIntegralInRange<uint32_t>(2, 4);
                        switchConfig.includeDefault = provider.ConsumeBool();
                        switchConfig.allCasesBreak = true; // Always use break
                        nestedSpec.switchConfig = switchConfig;
                    }
                    break;
            }
        } else {
            // Inside a loop, we can have another loop or an if
            int choice = provider.ConsumeIntegralInRange<int>(0, 3);
            switch (choice) {
                case 0:
                    nestedSpec.type = BlockSpec::FOR_LOOP;
                    break;
                case 1:
                    nestedSpec.type = BlockSpec::WHILE_LOOP;
                    break;
                case 2:
                    nestedSpec.type = BlockSpec::IF;
                    break;
                case 3:
                    nestedSpec.type = BlockSpec::IF_ELSE;
                    break;
            }
        }
    } else if (parentSpec.type == BlockSpec::SWITCH_STMT) {
        // Inside a switch case, we can have loops or if statements
        int choice = provider.ConsumeIntegralInRange<int>(0, 4);
        switch (choice) {
            case 0:
                nestedSpec.type = BlockSpec::FOR_LOOP;
                break;
            case 1:
                nestedSpec.type = BlockSpec::WHILE_LOOP;
                break;
            case 2:
                nestedSpec.type = BlockSpec::IF;
                break;
            case 3:
                nestedSpec.type = BlockSpec::IF_ELSE;
                break;
            case 4:
                // Another nested switch, but only if we have depth left
                if (nestedSpec.nestingContext.currentDepth < nestedSpec.nestingContext.maxDepth - 1) {
                    nestedSpec.type = BlockSpec::SWITCH_STMT;
                } else {
                    nestedSpec.type = BlockSpec::IF;
                }
                break;
        }
    } else {
        // Inside an if, we can have a loop, switch, or another if
        int choice = provider.ConsumeIntegralInRange<int>(0, 3);
        switch (choice) {
            case 0:
                nestedSpec.type = BlockSpec::FOR_LOOP;
                break;
            case 1:
                nestedSpec.type = BlockSpec::WHILE_LOOP;
                break;
            case 2:
                nestedSpec.type = BlockSpec::IF;
                break;
            case 3:
                nestedSpec.type = BlockSpec::SWITCH_STMT;
                break;
        }
    }
    
    // Create pattern for nested block
    nestedSpec.pattern = createRandomPattern(provider);
    
    // Configure switch-specific settings if needed
    if (nestedSpec.type == BlockSpec::SWITCH_STMT) {
        BlockSpec::SwitchConfig switchConfig;
        switchConfig.selectorType = static_cast<BlockSpec::SwitchConfig::SelectorType>(
            provider.ConsumeIntegralInRange<int>(0, 2));
        switchConfig.numCases = provider.ConsumeIntegralInRange<uint32_t>(2, 4);
        switchConfig.includeDefault = provider.ConsumeBool();
        switchConfig.allCasesBreak = true; // Always use break to avoid fall-through complexity
        nestedSpec.switchConfig = switchConfig;
    }
    
    // Handle break/continue based on context
    bool hasLoop = hasEnclosingLoop(nestedSpec.nestingContext);
    bool hasSwitch = hasEnclosingSwitch(nestedSpec.nestingContext);
    
    if (nestedSpec.type == BlockSpec::FOR_LOOP || 
        nestedSpec.type == BlockSpec::WHILE_LOOP) {
        // Loops can always have break/continue
        nestedSpec.includeBreak = provider.ConsumeBool();
        nestedSpec.includeContinue = provider.ConsumeBool();
    } else {
        // If statements can have break if inside a loop or switch
        // Continue only if inside a loop
        nestedSpec.includeBreak = (hasLoop || hasSwitch) && provider.ConsumeBool();
        nestedSpec.includeContinue = hasLoop && provider.ConsumeBool();
    }
    
    return nestedSpec;
}

std::string ControlFlowGenerator::generateLoopVariable(
    const BlockSpec::NestingContext& context,
    BlockSpec::Type loopType,
    ProgramState& state) {
    
    std::string varName;
    
    if (loopType == BlockSpec::FOR_LOOP) {
        // For nested for loops: i0, i1, i2...
        varName = "i" + std::to_string(state.nextVarIndex++);
    } else {
        // For while loops: counter0, counter1...
        varName = "counter" + std::to_string(state.nextVarIndex++);
    }
    
    state.declaredVariables.insert(varName);
    return varName;
}

std::vector<std::unique_ptr<interpreter::Statement>>
ControlFlowGenerator::generateNestedBody(const BlockSpec& spec, ProgramState& state,
                                        FuzzedDataProvider& provider) {
    std::vector<std::unique_ptr<interpreter::Statement>> body;
    
    // Decision 1: Add wave operation at current level?
    // 70% chance to add wave operation
    if (provider.ConsumeIntegralInRange<uint32_t>(0, 9) < 7) {
        // Generate wave operation at this level
        uint32_t waveSize = state.program.waveSize > 0 ? state.program.waveSize : 32;
        auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
        
        std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
        auto waveOp = generateWaveOperation(state, provider);
        
        // Use += for accumulation to see cumulative effects
        auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
        auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(currentResult),
            std::move(waveOp),
            interpreter::BinaryOpExpr::Add
        );
        waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(addExpr)
        ));
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(participantCondition),
            std::move(waveBody)
        ));
    }
    
    // Decision 2: Add nested control flow?
    // Heavily bias towards nesting (90% chance)
    if (spec.nestingContext.allowNesting && 
        spec.nestingContext.currentDepth < spec.nestingContext.maxDepth &&
        provider.ConsumeIntegralInRange<uint32_t>(0, 9) < 9) {  // 90% chance
        
        // Generate nested control flow
        BlockSpec nestedSpec = createNestedBlockSpec(spec, state, provider);
        auto nestedStatements = generateBlock(nestedSpec, state, provider);
        
        for (auto& stmt : nestedStatements) {
            body.push_back(std::move(stmt));
        }
    }
    
    // Decision 3: Add another wave operation after nesting?
    // 60% chance to add another wave operation
    if (provider.ConsumeIntegralInRange<uint32_t>(0, 9) < 6) {
        uint32_t waveSize = state.program.waveSize > 0 ? state.program.waveSize : 32;
        auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
        
        std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
        auto waveOp = generateWaveOperation(state, provider);
        
        // Use += for accumulation to see cumulative effects
        auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
        auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(currentResult),
            std::move(waveOp),
            interpreter::BinaryOpExpr::Add
        );
        waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(addExpr)
        ));
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(participantCondition),
            std::move(waveBody)
        ));
    }
    
    // Ensure we have at least one statement
    if (body.empty()) {
        // Fallback: generate at least one wave operation
        uint32_t waveSize = state.program.waveSize > 0 ? state.program.waveSize : 32;
        auto participantCondition = spec.pattern->generateCondition(waveSize, provider);
        
        std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
        auto waveOp = generateWaveOperation(state, provider);
        
        // Use += for accumulation to see cumulative effects
        auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
        auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(currentResult),
            std::move(waveOp),
            interpreter::BinaryOpExpr::Add
        );
        waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(addExpr)
        ));
        
        body.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(participantCondition),
            std::move(waveBody)
        ));
    }
    
    return body;
}

std::unique_ptr<interpreter::Expression>
ControlFlowGenerator::generateWaveOperation(ProgramState& state, FuzzedDataProvider& provider) {
    // Choose operation type (excluding Product to avoid overflow)
    interpreter::WaveActiveOp::OpType opType;
    uint32_t opChoice = provider.ConsumeIntegralInRange<uint32_t>(0, 2);
    switch (opChoice) {
        case 0: opType = interpreter::WaveActiveOp::Sum; break;
        case 1: opType = interpreter::WaveActiveOp::Min; break;
        case 2: opType = interpreter::WaveActiveOp::Max; break;
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
        // Use lane-based expression or literal
        uint32_t choice = provider.ConsumeIntegralInRange<uint32_t>(0, 2);
        switch (choice) {
            case 0:
                // Use lane index directly
                input = std::make_unique<interpreter::LaneIndexExpr>();
                break;
            case 1:
                // Use lane index with some arithmetic
                input = std::make_unique<interpreter::BinaryOpExpr>(
                    std::make_unique<interpreter::LaneIndexExpr>(),
                    std::make_unique<interpreter::LiteralExpr>(
                        provider.ConsumeIntegralInRange<int>(1, 5)),
                    interpreter::BinaryOpExpr::Add
                );
                break;
            default:
                // Use literal
                input = std::make_unique<interpreter::LiteralExpr>(
                    provider.ConsumeIntegralInRange<int>(1, 10));
                break;
        }
    }
    
    return std::make_unique<interpreter::WaveActiveOp>(std::move(input), opType);
}

std::unique_ptr<interpreter::Statement>
ControlFlowGenerator::generateSwitch(const BlockSpec& spec, ProgramState& state,
                                    FuzzedDataProvider& provider) {
    // Ensure switch config is present
    if (!spec.switchConfig.has_value()) {
        // Fallback: create default switch config
        BlockSpec::SwitchConfig defaultConfig;
        defaultConfig.selectorType = BlockSpec::SwitchConfig::LANE_MODULO;
        defaultConfig.numCases = 3;
        defaultConfig.includeDefault = false;
        defaultConfig.allCasesBreak = true;
        
        // Cannot copy BlockSpec due to unique_ptr, just set the config directly
        const_cast<BlockSpec&>(spec).switchConfig = defaultConfig;
        return generateSwitch(spec, state, provider);
    }
    
    const auto& config = spec.switchConfig.value();
    
    // Step 1: Generate switch condition based on selector type
    std::unique_ptr<interpreter::Expression> condition;
    
    switch (config.selectorType) {
    case BlockSpec::SwitchConfig::LANE_MODULO:
        // laneId % numCases
        condition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int>(config.numCases)),
            interpreter::BinaryOpExpr::Mod
        );
        break;
        
    case BlockSpec::SwitchConfig::THREAD_ID_BASED:
        // Use WaveGetLaneIndex() instead of tid.x for better wave operation support
        condition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int>(config.numCases)),
            interpreter::BinaryOpExpr::Mod
        );
        break;
        
    case BlockSpec::SwitchConfig::VARIABLE_BASED:
        // Use existing variable or create a simple one
        if (!state.declaredVariables.empty() && provider.ConsumeBool()) {
            // Filter for integer variables that could be used as selector
            std::vector<std::string> validVars;
            for (const auto& var : state.declaredVariables) {
                // Skip loop counters and result variable
                if (var != "result" && var.substr(0, 1) != "i" && var.substr(0, 7) != "counter") {
                    validVars.push_back(var);
                }
            }
            
            if (!validVars.empty()) {
                size_t idx = provider.ConsumeIntegralInRange<size_t>(0, validVars.size() - 1);
                condition = std::make_unique<interpreter::BinaryOpExpr>(
                    std::make_unique<interpreter::VariableExpr>(validVars[idx]),
                    std::make_unique<interpreter::LiteralExpr>(static_cast<int>(config.numCases)),
                    interpreter::BinaryOpExpr::Mod
                );
                break;
            }
        }
        // Fallback to lane modulo
        condition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(static_cast<int>(config.numCases)),
            interpreter::BinaryOpExpr::Mod
        );
        break;
    }
    
    // Step 2: Create switch statement
    auto switchStmt = std::make_unique<interpreter::SwitchStmt>(std::move(condition));
    
    // Step 3: Generate cases
    for (uint32_t i = 0; i < config.numCases; ++i) {
        std::vector<std::unique_ptr<interpreter::Statement>> caseBody;
        
        // Decide whether to add nested control flow (50% chance)
        bool addNestedControlFlow = provider.ConsumeBool() && spec.nestingContext.allowNesting;
        
        if (addNestedControlFlow && spec.nestingContext.currentDepth < spec.nestingContext.maxDepth) {
            // Create a nested block spec for this case
            BlockSpec caseBlockSpec = createNestedBlockSpec(spec, state, provider);
            
            // Override type selection for variety - allow for loops and other structures
            int nestedChoice = provider.ConsumeIntegralInRange<int>(0, 4);
            switch (nestedChoice) {
                case 0:
                    caseBlockSpec.type = BlockSpec::FOR_LOOP;
                    break;
                case 1:
                    caseBlockSpec.type = BlockSpec::WHILE_LOOP;
                    break;
                case 2:
                    caseBlockSpec.type = BlockSpec::IF;
                    break;
                case 3:
                    caseBlockSpec.type = BlockSpec::IF_ELSE;
                    break;
                case 4:
                    // Nested switch - but avoid too deep nesting
                    if (spec.nestingContext.currentDepth < spec.nestingContext.maxDepth - 1) {
                        caseBlockSpec.type = BlockSpec::SWITCH_STMT;
                    } else {
                        caseBlockSpec.type = BlockSpec::IF;
                    }
                    break;
            }
            
            // Generate the nested control flow
            auto nestedStatements = generateBlock(caseBlockSpec, state, provider);
            for (auto& stmt : nestedStatements) {
                caseBody.push_back(std::move(stmt));
            }
        } else {
            // Original behavior: just add wave operation
            // Create case-specific participation pattern
            std::unique_ptr<interpreter::Expression> participantCondition;
            
            // Different patterns for different cases
            switch (i % 3) {
            case 0:
                // Threshold pattern: first N lanes
                participantCondition = std::make_unique<interpreter::BinaryOpExpr>(
                    std::make_unique<interpreter::LaneIndexExpr>(),
                    std::make_unique<interpreter::LiteralExpr>(static_cast<int>(8 + i * 4)),
                    interpreter::BinaryOpExpr::Lt
                );
                break;
                
            case 1:
                // Modulo pattern: every Nth lane
                participantCondition = std::make_unique<interpreter::BinaryOpExpr>(
                    std::make_unique<interpreter::BinaryOpExpr>(
                        std::make_unique<interpreter::LaneIndexExpr>(),
                        std::make_unique<interpreter::LiteralExpr>(2),
                        interpreter::BinaryOpExpr::Mod
                    ),
                    std::make_unique<interpreter::LiteralExpr>(0),
                    interpreter::BinaryOpExpr::Eq
                );
                break;
                
            case 2:
                // All lanes in this case
                participantCondition = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
                break;
            }
            
            // Add wave operation with case-specific input
            std::vector<std::unique_ptr<interpreter::Statement>> waveBody;
            auto waveInput = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(i + 1));
            auto waveOp = std::make_unique<interpreter::WaveActiveOp>(
                std::move(waveInput), 
                interpreter::WaveActiveOp::Sum
            );
            
            // Accumulate result
            auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
            auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
                std::move(currentResult),
                std::move(waveOp),
                interpreter::BinaryOpExpr::Add
            );
            waveBody.push_back(std::make_unique<interpreter::AssignStmt>(
                "result",
                std::move(addExpr)
            ));
            
            // Wrap wave operation in participation condition
            caseBody.push_back(std::make_unique<interpreter::IfStmt>(
                std::move(participantCondition),
                std::move(waveBody)
            ));
        }
        
        // Add break statement if configured
        if (config.allCasesBreak) {
            caseBody.push_back(std::make_unique<interpreter::BreakStmt>());
        }
        
        switchStmt->addCase(i, std::move(caseBody));
    }
    
    // Step 4: Optional default case
    if (config.includeDefault) {
        std::vector<std::unique_ptr<interpreter::Statement>> defaultBody;
        
        // Simple wave operation for default case
        auto defaultWaveOp = std::make_unique<interpreter::WaveActiveOp>(
            std::make_unique<interpreter::LiteralExpr>(99),  // Distinctive value
            interpreter::WaveActiveOp::Sum
        );
        
        auto currentResult = std::make_unique<interpreter::VariableExpr>("result");
        auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(currentResult),
            std::move(defaultWaveOp),
            interpreter::BinaryOpExpr::Add
        );
        defaultBody.push_back(std::make_unique<interpreter::AssignStmt>(
            "result",
            std::move(addExpr)
        ));
        
        if (config.allCasesBreak) {
            defaultBody.push_back(std::make_unique<interpreter::BreakStmt>());
        }
        
        switchStmt->addDefault(std::move(defaultBody));
    }
    
    return switchStmt;
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
    
    FUZZER_DEBUG_LOG("\naddStatementsToProgram: offset=" << offset << ", size=" << size << ", available data=" << (size - offset) << "\n");
    
    // Add one round of new statements to the existing program
    GenerationRound roundInfo;
    roundInfo.roundNumber = state.history.size();
    
    // Generate new control flow block
    auto pattern = createPattern(provider);
    auto blockType = static_cast<ControlFlowGenerator::BlockSpec::Type>(
        provider.ConsumeIntegralInRange<int>(0, 9) < 4 ? provider.ConsumeIntegralInRange<int>(0, 5) : 6);  // 40% chance for SWITCH_STMT
    
    // Debug: Print selected block type
    const char* blockTypeNames[] = {"IF", "IF_ELSE", "NESTED_IF", "FOR_LOOP", "WHILE_LOOP", "CASCADING_IF_ELSE", "SWITCH_STMT"};
    FUZZER_DEBUG_LOG("addStatements: Selected block type: " << blockTypeNames[blockType] << " (" << blockType << ")\n");
    
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
            provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3,
        std::nullopt,  // switchConfig - will be set below if needed
        {} // nestingContext with default initialization
    };
    
    // Configure switch-specific settings if it's a switch statement
    if (blockType == ControlFlowGenerator::BlockSpec::SWITCH_STMT) {
        ControlFlowGenerator::BlockSpec::SwitchConfig switchConfig;
        
        // Randomly select selector type
        uint32_t selectorChoice = provider.ConsumeIntegralInRange<uint32_t>(0, 2);
        switchConfig.selectorType = static_cast<ControlFlowGenerator::BlockSpec::SwitchConfig::SelectorType>(selectorChoice);
        
        // Number of cases (2-4)
        switchConfig.numCases = provider.ConsumeIntegralInRange<uint32_t>(2, 4);
        
        // 30% chance to include default case
        switchConfig.includeDefault = provider.ConsumeIntegralInRange<uint32_t>(0, 9) < 3;
        
        // For phase 1, always use break statements
        switchConfig.allCasesBreak = true;
        
        spec.switchConfig = switchConfig;
    }
    
    // Enable nesting for this block - always allow nesting to maximize complexity
    spec.nestingContext.allowNesting = true;  // Always allow nesting
    spec.nestingContext.maxDepth = provider.ConsumeIntegralInRange<uint32_t>(2, 3);  // Reduced max depth to avoid timeout
    spec.nestingContext.currentDepth = 0;
    
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