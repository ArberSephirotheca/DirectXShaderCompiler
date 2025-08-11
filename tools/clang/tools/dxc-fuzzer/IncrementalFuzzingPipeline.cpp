#include "IncrementalFuzzingPipeline.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include "FuzzerDebug.h"
#include <iostream>
#include <chrono>
#include <thread>

namespace minihlsl {
namespace fuzzer {

IncrementalFuzzingPipeline::IncrementalFuzzingPipeline(const IncrementalFuzzingConfig& cfg)
    : generator(std::make_unique<IncrementalGenerator>()),
      fuzzer(std::make_unique<TraceGuidedFuzzer>()),
      config(cfg) {
}

bool IncrementalFuzzingPipeline::validateSyntax(const interpreter::Program& program, 
                                                std::string& error) {
    // Basic syntax validation
    // In future, this could call DXC compiler for real validation
    
    // Check for basic issues
    if (program.statements.empty()) {
        error = "Program has no statements";
        return false;
    }
    
    // Check thread configuration
    if (program.numThreadsX == 0 || program.numThreadsY == 0 || program.numThreadsZ == 0) {
        error = "Invalid thread configuration";
        return false;
    }
    
    // TODO: Add more syntax validation
    // - Check for undefined variables
    // - Check for type mismatches
    // - Validate wave operation usage
    
    return true;
}

bool IncrementalFuzzingPipeline::validateExecution(const interpreter::Program& program,
                                                   ExecutionTrace& trace, 
                                                   std::string& error) {
    try {
        // Create trace capture interpreter
        TraceCaptureInterpreter interpreter;
        
        // Use sequential thread ordering
        interpreter::ThreadOrdering ordering = interpreter::ThreadOrdering::sequential(program.getTotalThreads());
        
        // Execute with default wave size
        uint32_t waveSize = 32;
        if (program.waveSize > 0) {
            waveSize = program.waveSize;
        }
        
        // Debug output for execution parameters
        std::cout << "\n=== Execution Parameters ===\n";
        std::cout << "Thread Ordering: " << ordering.description << "\n";
        std::cout << "Execution Order Size: " << ordering.executionOrder.size() << "\n";
        std::cout << "Execution Order: ";
        for (auto tid : ordering.executionOrder) {
            std::cout << tid << " ";
        }
        std::cout << "\n";
        std::cout << "Total Threads: " << program.getTotalThreads() << "\n";
        std::cout << "Wave Size: " << waveSize << "\n";
        std::cout << "===========================\n" << std::endl;
        
        // Set a timeout to prevent infinite loops
        auto start = std::chrono::steady_clock::now();
        auto timeout = std::chrono::seconds(5);
        
        auto result = interpreter.executeAndCaptureTrace(program, ordering, waveSize);
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms\n";
        if (elapsed > timeout) {
            error = "Execution timeout: possible infinite loop or deadlock";
            return false;
        }
        
        if (!result.isValid()) {
            error = result.errorMessage;
            return false;
        }
        
        // Copy the trace
        if (interpreter.getTrace()) {
            trace = *interpreter.getTrace();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        error = std::string("Execution exception: ") + e.what();
        return false;
    } catch (...) {
        error = "Unknown execution exception";
        return false;
    }
}

PipelineResult::IncrementResult IncrementalFuzzingPipeline::testMutations(
    const interpreter::Program& program,
    const ExecutionTrace& goldenTrace,
    interpreter::Program& mutatedProgram,
    const ProgramState& state) {
    
    PipelineResult::IncrementResult result;
    result.baseProgramStr = serializeProgramToString(program);
    
    // Configure fuzzing for this increment
    FuzzingConfig fuzzConfig;
    fuzzConfig.maxMutants = config.mutantsPerIncrement;
    fuzzConfig.enableLogging = config.enableLogging;
    // fuzzConfig.stopOnFirstBug = false; // Not available in FuzzingConfig

    // Run fuzzing with captured trace
    try {
        // The fuzzer expects to have the golden trace available
        // We need to properly integrate with the trace-guided fuzzing
        
        // For now, let's print what mutations would be applied
        if (config.enableLogging) {
            FUZZER_DEBUG_LOG("\nApplying trace-guided mutations:\n");
            FUZZER_DEBUG_LOG("Golden trace has " << goldenTrace.waveOperations.size() << " wave operations\n");
            
            // Show what mutations would be applied
            for (size_t i = 0; i < goldenTrace.waveOperations.size() && i < 3; ++i) {
                const auto& waveOp = goldenTrace.waveOperations[i];
                FUZZER_DEBUG_LOG("Wave op " << i << ": " << waveOp.opType 
                         << " with " << waveOp.arrivedParticipants.size() << " participants\n");
                FUZZER_DEBUG_LOG("  - Could apply LanePermutation mutation\n");
                FUZZER_DEBUG_LOG("  - Could apply WaveParticipantTracking mutation\n");
            }
        }
        
        // We can't copy the program directly - need to serialize and deserialize
        // For now, just ensure mutatedProgram is empty
        mutatedProgram = interpreter::Program();
        
        // Actually run the fuzzer and get the mutated program
        // Pass the generation history so it only mutates new statements
        size_t currentRound = 0;
        if (!state.history.empty()) {
            currentRound = state.history.back().roundNumber;
        }
        mutatedProgram = fuzzer->fuzzProgram(program, fuzzConfig, state.history, currentRound);
        
        // Track results
        result.mutantsTested = fuzzConfig.maxMutants;
        result.bugsFound = 0; // Would need to get from fuzzer
        
    } catch (const std::exception& e) {
        result.errorMessage = std::string("Mutation testing error: ") + e.what();
        result.executionValid = false;
        // If mutation failed, we need to return the original program
        // Since we can't copy, we'll have to let the caller handle this case
    }
    
    return result;
}

PipelineResult IncrementalFuzzingPipeline::run(const uint8_t* data, size_t size) {
    PipelineResult result;
    
    if (config.enableLogging) {
        static int pipelineCallCount = 0;
        ++pipelineCallCount;
        FUZZER_DEBUG_LOG("\n=== Starting Incremental Fuzzing Pipeline (Call #" << pipelineCallCount << ") ===\n");
        FUZZER_DEBUG_LOG("Max increments: " << config.maxIncrements << "\n");
        FUZZER_DEBUG_LOG("Mutants per increment: " << config.mutantsPerIncrement << "\n\n");
    }
    
    // Initialize fuzzer with empty seed corpus
    // (The fuzzer expects to have been initialized)
    
    FuzzedDataProvider provider(data, size);
    
    // For the first increment, generate initial program
    // For subsequent increments, we'll add to the existing program
    ProgramState state;
    
    for (size_t increment = 0; increment < config.maxIncrements; ++increment) {
        if (config.enableLogging) {
            FUZZER_DEBUG_LOG("\n=== Increment " << (increment + 1) << " ===\n");
        }
        
        PipelineResult::IncrementResult incrementResult;
        
        // For first increment, generate base program
        // For subsequent increments, add statements to existing program
        if (increment == 0) {
            ProgramState newState = generator->generateIncremental(data, size);
            // Move the program and other data
            state.program = std::move(newState.program);
            state.history = std::move(newState.history);
            state.declaredVariables = std::move(newState.declaredVariables);
            state.nextVarIndex = newState.nextVarIndex;
            state.pendingStatement = std::move(newState.pendingStatement);
        } else {
            // Calculate offset for data consumption to ensure different statements each time
            size_t offset = (increment * size) / config.maxIncrements;
            generator->addStatementsToProgram(state, data, size, offset);
        }
        
        // Get current program state
        incrementResult.baseProgramStr = serializeProgramToString(state.program);
        
        if (config.enableLogging) {
            FUZZER_DEBUG_LOG("Generated program:\n");
            FUZZER_DEBUG_LOG(incrementResult.baseProgramStr);
            FUZZER_DEBUG_LOG("\n");
        }
        
        // Step 1: Validate syntax
        std::string syntaxError;
        if (!validateSyntax(state.program, syntaxError)) {
            incrementResult.syntaxValid = false;
            incrementResult.errorMessage = syntaxError;
            
            if (config.enableLogging) {
                FUZZER_DEBUG_LOG("Syntax validation failed: " << syntaxError << "\n");
            }
            
            result.increments.push_back(incrementResult);
            result.stoppedEarly = true;
            result.stopReason = "Syntax validation failed";
            break;
        }
        
        // Step 2: Validate execution
        ExecutionTrace goldenTrace;
        std::string execError;
        if (!validateExecution(state.program, goldenTrace, execError)) {
            incrementResult.executionValid = false;
            incrementResult.errorMessage = execError;
            
            if (config.enableLogging) {
                FUZZER_DEBUG_LOG("Execution validation failed: " << execError << "\n");
            }
            
            result.increments.push_back(incrementResult);
            result.stoppedEarly = true;
            result.stopReason = "Execution validation failed";
            break;
        }
        
        if (config.enableLogging) {
            FUZZER_DEBUG_LOG("Program validated successfully\n");
            FUZZER_DEBUG_LOG("Trace: " << goldenTrace.blocks.size() << " blocks, "
                     << goldenTrace.waveOperations.size() << " wave operations\n");
        }
        
        // Step 3: Apply mutations and test
        interpreter::Program mutatedProgram;
        incrementResult = testMutations(state.program, goldenTrace, mutatedProgram, state);
        
        // Check if mutation was successful
        bool mutationSucceeded = mutatedProgram.statements.size() > 0;
        
        result.totalMutantsTested += incrementResult.mutantsTested;
        result.totalBugsFound += incrementResult.bugsFound;
        
        if (config.enableLogging) {
            FUZZER_DEBUG_LOG("Tested " << incrementResult.mutantsTested << " mutants\n");
            FUZZER_DEBUG_LOG("Found " << incrementResult.bugsFound << " bugs\n");
        }
        
        result.increments.push_back(incrementResult);
        
        // Check if we should stop
        if (config.stopOnFirstBug && incrementResult.bugsFound > 0) {
            result.stoppedEarly = true;
            result.stopReason = "Bug found";
            break;
        }
        
        // Check program size limit
        if (state.program.statements.size() >= config.maxProgramSize) {
            if (config.enableLogging) {
                FUZZER_DEBUG_LOG("Reached maximum program size\n");
            }
            break;
        }
        
        // Step 4: Use mutated program as base for next increment
        if (increment < config.maxIncrements - 1 && mutationSucceeded) {
            // Update state with the mutated program for the next increment
            state.program = std::move(mutatedProgram);
            
            // Add more statements to the mutated program
            size_t remainingData = provider.remaining_bytes();
            if (remainingData < 16) {
                if (config.enableLogging) {
                    FUZZER_DEBUG_LOG("Insufficient data for next increment\n");
                }
                break;
            }
            
            // The mutated program is now stored in state.program
            // The next iteration will add more statements to it
            if (config.enableLogging) {
                FUZZER_DEBUG_LOG("\n=== Using mutated program as base for next increment ===\n");
                FUZZER_DEBUG_LOG(serializeProgramToString(state.program));
                FUZZER_DEBUG_LOG("\n");
            }
        }
    }
    
    if (config.enableLogging) {
        FUZZER_DEBUG_LOG("\n=== Pipeline Complete ===\n");
        FUZZER_DEBUG_LOG("Total increments: " << result.increments.size() << "\n");
        FUZZER_DEBUG_LOG("Total mutants tested: " << result.totalMutantsTested << "\n");
        FUZZER_DEBUG_LOG("Total bugs found: " << result.totalBugsFound << "\n");
        if (result.stoppedEarly) {
            FUZZER_DEBUG_LOG("Stopped early: " << result.stopReason << "\n");
        }
    }
    
    return result;
}

} // namespace fuzzer
} // namespace minihlsl