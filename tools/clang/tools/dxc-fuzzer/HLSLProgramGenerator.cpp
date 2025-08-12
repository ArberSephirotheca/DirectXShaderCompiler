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
                provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3,
            {} // nestingContext with default initialization
        };
        
        // Enable nesting for this block
        spec.nestingContext.allowNesting = provider.ConsumeBool();
        spec.nestingContext.maxDepth = provider.ConsumeIntegralInRange<uint32_t>(2, 3);
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
        // This will only be true if hasEnclosingLoop was true when spec was created
        std::vector<std::unique_ptr<interpreter::Statement>> breakBody;
        breakBody.push_back(std::make_unique<interpreter::BreakStmt>());
        
        // Wrap break in another condition for variety
        auto breakCondition = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::LaneIndexExpr>(),
            std::make_unique<interpreter::LiteralExpr>(16),
            interpreter::BinaryOpExpr::Gt
        );
        
        thenBody.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(breakCondition),
            std::move(breakBody)
        ));
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
    } else {
        // Inside an if, we can have a loop
        nestedSpec.type = provider.ConsumeBool() ? BlockSpec::FOR_LOOP : BlockSpec::WHILE_LOOP;
    }
    
    // Create pattern for nested block
    nestedSpec.pattern = createRandomPattern(provider);
    
    // Handle break/continue based on context
    bool hasLoop = hasEnclosingLoop(nestedSpec.nestingContext);
    
    if (nestedSpec.type == BlockSpec::FOR_LOOP || 
        nestedSpec.type == BlockSpec::WHILE_LOOP) {
        // Loops can always have break/continue
        nestedSpec.includeBreak = provider.ConsumeBool();
        nestedSpec.includeContinue = provider.ConsumeBool();
    } else {
        // If statements can only have break/continue if inside a loop
        nestedSpec.includeBreak = hasLoop && provider.ConsumeBool();
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
    if (provider.ConsumeBool()) {
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
    if (spec.nestingContext.allowNesting && 
        spec.nestingContext.currentDepth < spec.nestingContext.maxDepth &&
        provider.ConsumeBool()) {
        
        // Generate nested control flow
        BlockSpec nestedSpec = createNestedBlockSpec(spec, state, provider);
        auto nestedStatements = generateBlock(nestedSpec, state, provider);
        
        for (auto& stmt : nestedStatements) {
            body.push_back(std::move(stmt));
        }
    }
    
    // Decision 3: Add another wave operation after nesting?
    if (provider.ConsumeBool()) {
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
            provider.ConsumeIntegralInRange<uint32_t>(2, 5) : 3,
        {} // nestingContext with default initialization
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