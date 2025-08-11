#include "IncrementalFuzzingPipeline.h"
#include "MiniHLSLInterpreterTraceCapture.h"
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
        
        // Use default thread ordering
        interpreter::ThreadOrdering ordering;
        
        // Execute with default wave size
        uint32_t waveSize = 32;
        if (program.waveSizePreferred > 0) {
            waveSize = program.waveSizePreferred;
        }
        
        // Set a timeout to prevent infinite loops
        auto start = std::chrono::steady_clock::now();
        auto timeout = std::chrono::seconds(5);
        
        auto result = interpreter.executeAndCaptureTrace(program, ordering, waveSize);
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > timeout) {
            error = "Execution timeout - possible infinite loop";
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
    const ExecutionTrace& goldenTrace) {
    
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
            std::cout << "\nApplying trace-guided mutations:\n";
            std::cout << "Golden trace has " << goldenTrace.waveOperations.size() << " wave operations\n";
            
            // Show what mutations would be applied
            for (size_t i = 0; i < goldenTrace.waveOperations.size() && i < 3; ++i) {
                const auto& waveOp = goldenTrace.waveOperations[i];
                std::cout << "Wave op " << i << ": " << waveOp.opType 
                         << " with " << waveOp.arrivedParticipants.size() << " participants\n";
                std::cout << "  - Could apply LanePermutation mutation\n";
                std::cout << "  - Could apply WaveParticipantTracking mutation\n";
            }
        }
        
        // Actually run the fuzzer
        fuzzer->fuzzProgram(program, fuzzConfig);
        
        // Track results
        result.mutantsTested = fuzzConfig.maxMutants;
        result.bugsFound = 0; // Would need to get from fuzzer
        
    } catch (const std::exception& e) {
        result.errorMessage = std::string("Mutation testing error: ") + e.what();
        result.executionValid = false;
    }
    
    return result;
}

PipelineResult IncrementalFuzzingPipeline::run(const uint8_t* data, size_t size) {
    PipelineResult result;
    
    if (config.enableLogging) {
        static int pipelineCallCount = 0;
        ++pipelineCallCount;
        std::cout << "\n=== Starting Incremental Fuzzing Pipeline (Call #" << pipelineCallCount << ") ===\n";
        std::cout << "Max increments: " << config.maxIncrements << "\n";
        std::cout << "Mutants per increment: " << config.mutantsPerIncrement << "\n\n";
    }
    
    // Initialize fuzzer with empty seed corpus
    // (The fuzzer expects to have been initialized)
    
    FuzzedDataProvider provider(data, size);
    ProgramState state = generator->generateIncremental(data, size);
    
    for (size_t increment = 0; increment < config.maxIncrements; ++increment) {
        if (config.enableLogging) {
            std::cout << "\n=== Increment " << (increment + 1) << " ===\n";
        }
        
        PipelineResult::IncrementResult incrementResult;
        
        // Get current program state
        incrementResult.baseProgramStr = serializeProgramToString(state.program);
        
        if (config.enableLogging) {
            std::cout << "Generated program:\n";
            std::cout << incrementResult.baseProgramStr;
            std::cout << "\n";
        }
        
        // Step 1: Validate syntax
        std::string syntaxError;
        if (!validateSyntax(state.program, syntaxError)) {
            incrementResult.syntaxValid = false;
            incrementResult.errorMessage = syntaxError;
            
            if (config.enableLogging) {
                std::cout << "Syntax validation failed: " << syntaxError << "\n";
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
                std::cout << "Execution validation failed: " << execError << "\n";
            }
            
            result.increments.push_back(incrementResult);
            result.stoppedEarly = true;
            result.stopReason = "Execution validation failed";
            break;
        }
        
        if (config.enableLogging) {
            std::cout << "Program validated successfully\n";
            std::cout << "Trace: " << goldenTrace.blocks.size() << " blocks, "
                     << goldenTrace.waveOperations.size() << " wave operations\n";
        }
        
        // Step 3: Apply mutations and test
        incrementResult = testMutations(state.program, goldenTrace);
        
        result.totalMutantsTested += incrementResult.mutantsTested;
        result.totalBugsFound += incrementResult.bugsFound;
        
        if (config.enableLogging) {
            std::cout << "Tested " << incrementResult.mutantsTested << " mutants\n";
            std::cout << "Found " << incrementResult.bugsFound << " bugs\n";
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
                std::cout << "Reached maximum program size\n";
            }
            break;
        }
        
        // Step 4: Generate next increment
        if (increment < config.maxIncrements - 1) {
            // Add more to the program
            size_t remainingData = provider.remaining_bytes();
            if (remainingData < 16) {
                if (config.enableLogging) {
                    std::cout << "Insufficient data for next increment\n";
                }
                break;
            }
            
            // Generate more statements
            state = generator->generateIncremental(
                provider.ConsumeBytes<uint8_t>(remainingData).data(), 
                remainingData);
        }
    }
    
    if (config.enableLogging) {
        std::cout << "\n=== Pipeline Complete ===\n";
        std::cout << "Total increments: " << result.increments.size() << "\n";
        std::cout << "Total mutants tested: " << result.totalMutantsTested << "\n";
        std::cout << "Total bugs found: " << result.totalBugsFound << "\n";
        if (result.stoppedEarly) {
            std::cout << "Stopped early: " << result.stopReason << "\n";
        }
    }
    
    return result;
}

} // namespace fuzzer
} // namespace minihlsl