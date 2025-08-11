// fuzz_hlsl_incremental.cpp - libFuzzer entry point for incremental HLSL generation

#include "HLSLProgramGenerator.h"
#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

// Global verbosity flag - can be set via environment variable
static int g_verbosity = 0;

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
        // Generate program incrementally
        fuzzer::IncrementalGenerator generator;
        auto state = generator.generateIncremental(Data, Size);
        
        // Prepare for execution (adds any missing initialization)
        auto program = fuzzer::prepareProgramForExecution(std::move(state.program));
        
        // Print generated program if verbosity is enabled
        if (g_verbosity >= 2) {
            std::cerr << "=== Generated HLSL Program ===\n";
            std::cerr << fuzzer::serializeProgramToString(program) << std::endl;
            std::cerr << "=== End Program ===\n";
        }
        
        // Print mutation history if verbosity is high
        if (g_verbosity >= 3) {
            std::cerr << "=== Mutation History ===\n";
            for (const auto& round : state.history) {
                std::cerr << "Round " << round.roundNumber << ": ";
                std::cerr << round.addedStatementIndices.size() << " statements added";
                if (!round.appliedMutations.empty()) {
                    std::cerr << ", mutations: ";
                    for (auto mut : round.appliedMutations) {
                        switch (mut) {
                            case fuzzer::MutationType::LanePermutation:
                                std::cerr << "LanePermutation ";
                                break;
                            case fuzzer::MutationType::ParticipantTracking:
                                std::cerr << "ParticipantTracking ";
                                break;
                            default:
                                std::cerr << "Unknown ";
                        }
                    }
                }
                std::cerr << "\n";
            }
            std::cerr << "=== End Mutation History ===\n";
        }
        
        // Create interpreter
        interpreter::MiniHLSLInterpreter interpreter;
        uint32_t waveSize = program.getEffectiveWaveSize(32);
        
        // Verify the program with multiple orderings
        auto verification = interpreter.verifyOrderIndependence(program, 10, waveSize);
        
        // Check for bugs
        if (!verification.isOrderIndependent) {
            // Log the bug
            std::cerr << "\n==== FOUND ORDER-DEPENDENT BEHAVIOR! ====\n";
            std::cerr << "Program:\n" << 
                fuzzer::serializeProgramToString(program) << std::endl;
            
            // Show which orderings differ
            std::cerr << "\nOrdering differences detected:\n";
            for (size_t i = 0; i < verification.orderings.size(); ++i) {
                std::cerr << "Ordering " << i << ": " 
                         << verification.orderings[i].description << "\n";
            }
            
            // Save to file
            saveBugToFile(Data, Size, program, "Order-dependent behavior");
            
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
                std::cerr << "\n==== FOUND WAVE OPERATION BUG! ====\n";
                saveBugToFile(Data, Size, program, "Wave operation bug");
                
                if (g_crashOnBug) {
                    abort();
                }
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

