#ifndef INCREMENTAL_FUZZING_PIPELINE_H
#define INCREMENTAL_FUZZING_PIPELINE_H

#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include "HLSLProgramGenerator.h"
#include <memory>
#include <vector>

namespace minihlsl {
namespace fuzzer {

// Configuration for the incremental fuzzing pipeline
struct IncrementalFuzzingConfig {
    size_t maxIncrements = 10;          // Maximum number of incremental generations
    size_t mutantsPerIncrement = 5;     // Number of mutants to test per increment
    size_t maxProgramSize = 100;        // Maximum statements in a program
    bool enableLogging = true;
    bool stopOnFirstBug = false;        // Stop pipeline on first bug found
    size_t randomSeed = 0;              // 0 = use random seed
};

// Result from one pipeline iteration
struct PipelineResult {
    struct IncrementResult {
        // Store serialized program instead of Program object to avoid copy issues
        std::string baseProgramStr;
        bool syntaxValid = true;
        bool executionValid = true;
        std::string errorMessage;
        
        // Mutation testing results
        size_t mutantsTested = 0;
        size_t bugsFound = 0;
        std::vector<BugReporter::BugReport> bugReports;
    };
    
    std::vector<IncrementResult> increments;
    size_t totalMutantsTested = 0;
    size_t totalBugsFound = 0;
    bool stoppedEarly = false;
    std::string stopReason;
};

class IncrementalFuzzingPipeline {
private:
    std::unique_ptr<IncrementalGenerator> generator;
    std::unique_ptr<TraceGuidedFuzzer> fuzzer;
    IncrementalFuzzingConfig config;
    
    // Validate program syntax (could integrate with DXC in future)
    bool validateSyntax(const interpreter::Program& program, std::string& error);
    
    // Execute program to check for interpreter errors
    bool validateExecution(const interpreter::Program& program, 
                          ExecutionTrace& trace, std::string& error);
    
    // Apply mutations and test
    // Returns the result and the mutated program via the mutatedProgram parameter
    PipelineResult::IncrementResult testMutations(
        const interpreter::Program& program,
        const ExecutionTrace& goldenTrace,
        interpreter::Program& mutatedProgram,
        const ProgramState& state,
        size_t increment);
    
public:
    IncrementalFuzzingPipeline(const IncrementalFuzzingConfig& cfg = {});
    
    // Run the full pipeline with given input data
    PipelineResult run(const uint8_t* data, size_t size);
    
    // Get current configuration
    const IncrementalFuzzingConfig& getConfig() const { return config; }
};

} // namespace fuzzer
} // namespace minihlsl

#endif // INCREMENTAL_FUZZING_PIPELINE_H