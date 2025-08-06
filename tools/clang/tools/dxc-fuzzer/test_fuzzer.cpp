// Simple test program to verify the fuzzer framework compiles

#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include <iostream>

int main() {
  std::cout << "Testing MiniHLSL Interpreter Fuzzer Framework\n";
  
  // Create a simple test program
  minihlsl::interpreter::Program program;
  program.numThreadsX = 32;
  program.numThreadsY = 1;
  program.numThreadsZ = 1;
  
  // Add a simple assignment: x = 1
  auto assign = std::make_unique<minihlsl::interpreter::AssignStmt>(
    "x",
    std::make_unique<minihlsl::interpreter::LiteralExpr>(minihlsl::interpreter::Value(1))
  );
  program.statements.push_back(std::move(assign));
  
  // Create fuzzer
  minihlsl::fuzzer::TraceGuidedFuzzer fuzzer;
  
  // Configure fuzzing
  minihlsl::fuzzer::FuzzingConfig config;
  config.threadgroupSize = 32;
  config.waveSize = 32;
  config.maxMutants = 10;
  config.enableLogging = true;
  
  std::cout << "Running fuzzer on test program...\n";
  
  try {
    fuzzer.fuzzProgram(program, config);
    std::cout << "Fuzzing completed successfully!\n";
  } catch (const std::exception& e) {
    std::cerr << "Error during fuzzing: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}