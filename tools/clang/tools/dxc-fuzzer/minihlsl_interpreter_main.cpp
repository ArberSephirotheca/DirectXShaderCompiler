#include "MiniHLSLInterpreter.h"
#include "MiniHLSLValidator.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <getopt.h>

// Clang headers for HLSL parsing
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
// Removed clang/Tooling/Tooling.h - not needed, using MiniHLSLValidator instead
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace minihlsl::interpreter;
using namespace clang;

struct Config {
    std::string inputFile;
    bool verbose = false;
    bool debugGraph = false;
    bool verifyOrder = true;
    uint32_t numOrderings = 10;
    uint32_t waveSize = 32;
    bool showHelp = false;
};

void printUsage(const char* progName) {
    std::cout << "MiniHLSL Interpreter - Standalone HLSL Program Executor\n"
              << "Usage: " << progName << " [options] <hlsl_file>\n\n"
              << "Options:\n"
              << "  -h, --help           Show this help message\n"
              << "  -v, --verbose        Enable verbose output\n"
              << "  -g, --debug-graph    Print dynamic execution graph\n"
              << "  -n, --no-verify      Skip order independence verification\n"
              << "  -o, --orderings N    Number of thread orderings to test (default: 10)\n"
              << "  -w, --wave-size N    Wave size (default: 32, must be power of 2)\n\n"
              << "Examples:\n"
              << "  " << progName << " wave_reduction.hlsl\n"
              << "  " << progName << " -v -g --wave-size 16 complex_shader.hlsl\n"
              << "  " << progName << " --no-verify simple_test.hlsl\n";
}

Config parseCommandLine(int argc, char* argv[]) {
    Config config;
    
    static struct option long_options[] = {
        {"help",        no_argument,       0, 'h'},
        {"verbose",     no_argument,       0, 'v'},
        {"debug-graph", no_argument,       0, 'g'},
        {"no-verify",   no_argument,       0, 'n'},
        {"orderings",   required_argument, 0, 'o'},
        {"wave-size",   required_argument, 0, 'w'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "hvgno:w:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                config.showHelp = true;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'g':
                config.debugGraph = true;
                break;
            case 'n':
                config.verifyOrder = false;
                break;
            case 'o':
                config.numOrderings = std::stoul(optarg);
                break;
            case 'w':
                config.waveSize = std::stoul(optarg);
                // Validate wave size is power of 2
                if (config.waveSize == 0 || (config.waveSize & (config.waveSize - 1)) != 0) {
                    std::cerr << "Error: Wave size must be a power of 2 (e.g., 8, 16, 32, 64)\n";
                    config.showHelp = true;
                }
                break;
            case '?':
                std::cerr << "Unknown option or missing argument.\n";
                config.showHelp = true;
                break;
        }
    }
    
    if (optind < argc && !config.showHelp) {
        config.inputFile = argv[optind];
    } else if (!config.showHelp) {
        std::cerr << "Error: No input file specified.\n";
        config.showHelp = true;
    }
    
    return config;
}

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    
    return content;
}

// Removed MainFunctionFinder and HLSLASTConsumer - now using MiniHLSLValidator

// Removed HLSLParseAction - now using MiniHLSLValidator

void printExecutionResult(const ExecutionResult& result, const std::string& description, bool verbose) {
    std::cout << "=== " << description << " ===\n";
    
    if (!result.isValid()) {
        std::cout << "❌ Execution failed: " << result.errorMessage << "\n";
        if (result.hasDataRace) {
            std::cout << "⚠️  Data race detected!\n";
        }
        return;
    }
    
    std::cout << "✅ Execution successful\n";
    
    // Print shared memory state
    if (!result.sharedMemoryState.empty()) {
        std::cout << "Shared Memory State:\n";
        for (const auto& [addr, val] : result.sharedMemoryState) {
            std::cout << "  [" << addr << "] = " << val.toString() << "\n";
        }
    }
    
    // Print return values
    if (!result.threadReturnValues.empty()) {
        std::cout << "Thread Return Values:\n";
        size_t displayCount = verbose ? result.threadReturnValues.size() : std::min(size_t(8), result.threadReturnValues.size());
        
        for (size_t i = 0; i < displayCount; ++i) {
            std::cout << "  Thread " << i << ": " << result.threadReturnValues[i].toString() << "\n";
        }
        
        if (!verbose && result.threadReturnValues.size() > 8) {
            std::cout << "  ... (" << (result.threadReturnValues.size() - 8) << " more threads)\n";
        }
    }
    
    std::cout << "\n";
}

void printVerificationResult(const MiniHLSLInterpreter::VerificationResult& verification, bool verbose) {
    std::cout << "=== Order Independence Verification ===\n";
    
    if (verification.isOrderIndependent) {
        std::cout << "✅ PASS: Program is order-independent!\n";
        std::cout << "Verified across " << verification.orderings.size() << " different thread orderings.\n";
    } else {
        std::cout << "❌ FAIL: Program is order-dependent!\n";
        std::cout << "Divergence detected:\n" << verification.divergenceReport << "\n";
    }
    
    if (verbose) {
        std::cout << "\nThread orderings tested:\n";
        for (size_t i = 0; i < verification.orderings.size(); ++i) {
            const auto& ordering = verification.orderings[i];
            std::cout << "  " << (i + 1) << ". " << ordering.description << "\n";
            
            if (ordering.executionOrder.size() <= 16) {
                std::cout << "     Order: ";
                for (size_t j = 0; j < ordering.executionOrder.size(); ++j) {
                    std::cout << ordering.executionOrder[j];
                    if (j < ordering.executionOrder.size() - 1) std::cout << " → ";
                }
                std::cout << "\n";
            }
        }
    }
    
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    Config config = parseCommandLine(argc, argv);
    
    if (config.showHelp) {
        printUsage(argv[0]);
        return config.inputFile.empty() ? 1 : 0;
    }
    
    std::cout << "MiniHLSL Interpreter\n";
    std::cout << "====================\n";
    std::cout << "Input file: " << config.inputFile << "\n";
    std::cout << "Wave size: " << config.waveSize << "\n";
    if (config.verifyOrder) {
        std::cout << "Order verification: " << config.numOrderings << " orderings\n";
    }
    std::cout << "\n";
    
    try {
        // Read HLSL source
        std::string hlslSource = readFile(config.inputFile);
        
        if (config.verbose) {
            std::cout << "=== HLSL Source ===\n";
            std::cout << hlslSource << "\n";
        }
        
        // Parse HLSL with MiniHLSLValidator
        minihlsl::MiniHLSLValidator validator;
        auto astResult = validator.validate_source_with_ast_ownership(hlslSource, config.inputFile);
        
        // we don't care about validaiton result for now
        // if (!astResult.is_valid()) {
        //     std::cerr << "❌ HLSL validation failed:\n";
        //     if (astResult.validation_result.is_err()) {
        //         for (const auto& error : astResult.validation_result.unwrap_err()) {
        //             std::cerr << "  " << error.message << "\n";
        //         }
        //     }
        //     return 1;
        // }
        
        auto* astContext = astResult.get_ast_context();
        auto* mainFunction = astResult.get_main_function();
        
        if (!astContext || !mainFunction) {
            std::cerr << "❌ Failed to get AST context or main function\n";
            return 1;
        }
        
        if (config.verbose) {
            std::cout << "✅ Found main function in HLSL source\n\n";
        }
        
        // Convert HLSL AST to interpreter program
        MiniHLSLInterpreter interpreter(42); // Fixed seed for reproducibility
        auto conversionResult = interpreter.convertFromHLSLAST(mainFunction, *astContext);
        
        if (!conversionResult.success) {
            std::cerr << "❌ Failed to convert HLSL to interpreter program: " 
                      << conversionResult.errorMessage << "\n";
            return 1;
        }
        
        if (config.verbose) {
            std::cout << "✅ Successfully converted HLSL to interpreter program\n";
            std::cout << "Program has " << conversionResult.program.statements.size() << " statements\n\n";
        }
        
        // Thread configuration is now determined solely by the HLSL [numthreads] attribute
        std::cout << "Thread configuration: [" << conversionResult.program.numThreadsX << ", "
                  << conversionResult.program.numThreadsY << ", " 
                  << conversionResult.program.numThreadsZ << "]\n\n";
        
        // Execute program with sequential ordering first
        auto sequentialResult = interpreter.execute(conversionResult.program, 
                                                   ThreadOrdering::sequential(conversionResult.program.getTotalThreads()),
                                                   config.waveSize);
        printExecutionResult(sequentialResult, "Sequential Execution", config.verbose);
        
        // Print dynamic execution graph if requested
        if (config.debugGraph) {
            // We would need access to the ThreadgroupContext to print the graph
            // For now, just indicate this feature is available
            std::cout << "=== Debug Graph Requested ===\n";
            std::cout << "Note: Debug graph printing requires execution context access.\n";
            std::cout << "This feature can be enabled by modifying the interpreter to expose ThreadgroupContext.\n\n";
        }
        
        // Verify order independence if requested
        if (config.verifyOrder) {
            auto verification = interpreter.verifyOrderIndependence(conversionResult.program, config.numOrderings, config.waveSize);
            printVerificationResult(verification, config.verbose);
        }
        
        std::cout << "✅ Execution completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}