// fuzz_hlsl_incremental.cpp - libFuzzer entry point for incremental HLSL generation

#include "HLSLProgramGenerator.h"
#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <memory>

// Global verbosity flag - can be set via environment variable
int g_verbosity = 0;

using namespace minihlsl;

// Global flag to control crash behavior
static bool g_crashOnBug = false;

// Helper to save bug to file
static void saveBugToFile(const uint8_t* data, size_t size, 
                         const interpreter::Program& program,
                         const std::string& description) {
    static int bugCounter = 0;
    
    // Create filename
    std::string filename = "bug_" + std::to_string(bugCounter++) + ".hlsl";
    
    // Write program to file
    std::ofstream out(filename);
    if (out.is_open()) {
        out << "// Bug: " << description << "\n";
        out << "// Fuzzer input size: " << size << " bytes\n\n";
        out << fuzzer::serializeProgramToString(program);
        out.close();
        
        std::cerr << "Bug saved to: " << filename << "\n";
    }
    
    // Also save the raw fuzzer input
    std::string inputFilename = "bug_" + std::to_string(bugCounter - 1) + ".input";
    std::ofstream inputOut(inputFilename, std::ios::binary);
    if (inputOut.is_open()) {
        inputOut.write(reinterpret_cast<const char*>(data), size);
        inputOut.close();
    }
}

// libFuzzer initialization
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    std::cerr << "HLSL Incremental Fuzzer initialized\n";
    std::cerr << "Generating programs with complex control flow and wave operations\n";
    
    // Check for verbosity in argv
    for (int i = 1; i < *argc; i++) {
        std::string arg((*argv)[i]);
        if (arg.find("-verbosity=") == 0) {
            g_verbosity = std::stoi(arg.substr(11));
        }
    }
    
    // Check environment variable for crash behavior
    if (getenv("FUZZ_CRASH_ON_BUG")) {
        g_crashOnBug = true;
        std::cerr << "Will crash on bug discovery (FUZZ_CRASH_ON_BUG set)\n";
    }
    
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Minimum size for meaningful generation
    if (Size < 16) return 0;
    
    try {
        // Generate base program WITHOUT mutations
        fuzzer::IncrementalGenerator generator;
        auto state = generator.generateIncremental(Data, Size);
        
        // Prepare for execution (adds any missing initialization)
        auto baseProgram = fuzzer::prepareProgramForExecution(std::move(state.program));
        
        // Print generated program if verbosity is enabled
        if (g_verbosity >= 2) {
            std::cerr << "=== Generated Base Program (No Mutations) ===\n";
            std::cerr << fuzzer::serializeProgramToString(baseProgram) << std::endl;
            std::cerr << "=== End Base Program ===\n";
        }
        
        // Step 1: Execute base program to get trace
        fuzzer::TraceCaptureInterpreter traceInterpreter;
        uint32_t waveSize = baseProgram.getEffectiveWaveSize(32);
        
        // Create a sequential ordering for initial execution
        interpreter::ThreadOrdering ordering = 
            interpreter::ThreadOrdering::sequential(baseProgram.getTotalThreads());
        
        // Execute and capture trace
        auto result = traceInterpreter.executeAndCaptureTrace(baseProgram, ordering, waveSize);
        
        if (!result.isValid()) {
            // Base program failed to execute - skip
            return 0;
        }
        
        const fuzzer::ExecutionTrace& baseTrace = *traceInterpreter.getTrace();
        
        if (g_verbosity >= 2) {
            std::cerr << "\n=== Base Program Execution Trace ===\n";
            std::cerr << "Wave operations: " << baseTrace.waveOperations.size() << "\n";
            std::cerr << "Blocks: " << baseTrace.blocks.size() << "\n";
            for (const auto& waveOp : baseTrace.waveOperations) {
                std::cerr << "  Wave op: " << waveOp.opType 
                          << " with " << waveOp.arrivedParticipants.size() 
                          << " participants in block " << waveOp.blockId << "\n";
            }
            std::cerr << "=== End Trace ===\n";
        }
        
        // Step 2: Apply mutations using the trace
        std::vector<std::unique_ptr<fuzzer::MutationStrategy>> mutationStrategies;
        mutationStrategies.push_back(std::make_unique<fuzzer::LanePermutationMutation>());
        mutationStrategies.push_back(std::make_unique<fuzzer::WaveParticipantTrackingMutation>());
        
        // Choose a mutation strategy
        FuzzedDataProvider provider(Data, Size);
        size_t strategyIdx = provider.ConsumeIntegralInRange<size_t>(0, mutationStrategies.size() - 1);
        auto* strategy = mutationStrategies[strategyIdx].get();
        
        if (g_verbosity >= 2) {
            std::cerr << "\n=== Applying Mutation Strategy: " << strategy->getName() << " ===\n";
        }
        
        // Generate mutants using the strategy and trace
        std::vector<interpreter::Program> mutants;
        
        // Try to apply mutation to each statement
        for (size_t i = 0; i < baseProgram.statements.size(); ++i) {
            const auto* stmt = baseProgram.statements[i].get();
            
            if (strategy->canApply(stmt, baseTrace)) {
                auto mutatedStmt = strategy->apply(stmt, baseTrace);
                if (mutatedStmt) {
                    // Create mutant program
                    interpreter::Program mutant;
                    mutant.numThreadsX = baseProgram.numThreadsX;
                    mutant.numThreadsY = baseProgram.numThreadsY;
                    mutant.numThreadsZ = baseProgram.numThreadsZ;
                    mutant.entryInputs = baseProgram.entryInputs;
                    mutant.globalBuffers = baseProgram.globalBuffers;
                    mutant.waveSizePreferred = baseProgram.waveSizePreferred;
                    mutant.waveSizeMin = baseProgram.waveSizeMin;
                    mutant.waveSizeMax = baseProgram.waveSizeMax;
                    
                    // Handle WaveParticipantTracking buffer requirements
                    if (dynamic_cast<fuzzer::WaveParticipantTrackingMutation*>(strategy)) {
                        // Check if _participant_check_sum buffer exists
                        bool hasBuffer = false;
                        for (const auto& buffer : mutant.globalBuffers) {
                            if (buffer.name == "_participant_check_sum") {
                                hasBuffer = true;
                                break;
                            }
                        }
                        
                        if (!hasBuffer) {
                            interpreter::GlobalBufferDecl participantBuffer;
                            participantBuffer.name = "_participant_check_sum";
                            participantBuffer.bufferType = "RWBuffer";
                            participantBuffer.elementType = interpreter::HLSLType::Uint;
                            participantBuffer.size = mutant.getTotalThreads();
                            participantBuffer.registerIndex = 1;
                            participantBuffer.isReadWrite = true;
                            mutant.globalBuffers.push_back(participantBuffer);
                            
                            // Add initialization
                            auto tidX = std::make_unique<interpreter::DispatchThreadIdExpr>(0);
                            auto zero = std::make_unique<interpreter::LiteralExpr>(0);
                            mutant.statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
                                "_participant_check_sum", std::move(tidX), std::move(zero)));
                        }
                    }
                    
                    // Clone all statements, replacing the mutated one
                    for (size_t j = 0; j < baseProgram.statements.size(); ++j) {
                        if (j == i) {
                            mutant.statements.push_back(std::move(mutatedStmt));
                        } else {
                            mutant.statements.push_back(baseProgram.statements[j]->clone());
                        }
                    }
                    
                    mutants.push_back(std::move(mutant));
                    
                    if (g_verbosity >= 2) {
                        std::cerr << "Applied mutation to statement " << i << "\n";
                    }
                    
                    // For now, just create one mutant
                    break;
                }
            }
        }
        
        // Step 3: Execute and verify mutants
        if (!mutants.empty()) {
            const auto& mutantProgram = mutants[0];
            
            if (g_verbosity >= 2) {
                std::cerr << "\n=== Mutant Program ===\n";
                std::cerr << fuzzer::serializeProgramToString(mutantProgram) << std::endl;
                std::cerr << "=== End Mutant Program ===\n";
            }
            
            // Create interpreter for verification
            interpreter::MiniHLSLInterpreter interpreter;
            
            // Verify the mutant program
            auto verification = interpreter.verifyOrderIndependence(mutantProgram, 10, waveSize);
            
            // Check for bugs
            if (!verification.isOrderIndependent) {
                // Log the bug
                std::cerr << "\n==== FOUND ORDER-DEPENDENT BEHAVIOR IN MUTANT! ====\n";
                std::cerr << "Base Program:\n" << 
                    fuzzer::serializeProgramToString(baseProgram) << std::endl;
                std::cerr << "\nMutant Program:\n" << 
                    fuzzer::serializeProgramToString(mutantProgram) << std::endl;
            
                // Show which orderings differ
                std::cerr << "\nOrdering differences detected:\n";
                for (size_t i = 0; i < verification.orderings.size(); ++i) {
                    std::cerr << "Ordering " << i << ": " 
                             << verification.orderings[i].description << "\n";
                }
                
                // Save to file
                saveBugToFile(Data, Size, mutantProgram, "Order-dependent mutant behavior");
                
                // Optionally crash for fuzzer to save the input
                if (g_crashOnBug) {
                    abort();
                }
            }
            
            // Additional checks for specific bugs
            if (!verification.results.empty()) {
                // Check for incorrect wave operation results
                bool hasWaveOpBug = false;
                for (const auto& result : verification.results) {
                    // This is a placeholder - real implementation would check
                    // for specific known incorrect patterns
                    if (!result.errorMessage.empty() && 
                        result.errorMessage.find("wave") != std::string::npos) {
                        hasWaveOpBug = true;
                        break;
                    }
                }
                
                if (hasWaveOpBug) {
                    std::cerr << "\n==== FOUND WAVE OPERATION BUG IN MUTANT! ====\n";
                    saveBugToFile(Data, Size, mutantProgram, "Wave operation bug in mutant");
                    
                    if (g_crashOnBug) {
                        abort();
                    }
                }
            }
        } else {
            // No mutants created
            if (g_verbosity >= 2) {
                std::cerr << "No mutants created - base program may not have wave operations\n";
            }
        }
        
    } catch (const std::exception& e) {
        // Ignore generation/execution errors - these are expected
        // during fuzzing as we explore edge cases
        return 0;
    }
    
    return 0;
}

// Optional: Custom mutator for better fuzzing
extern "C" size_t LLVMFuzzerCustomMutator(
    uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
    // For now, just use default mutation
    // In future, could implement structure-aware mutations
    return 0; // Return 0 to use default mutator
}

