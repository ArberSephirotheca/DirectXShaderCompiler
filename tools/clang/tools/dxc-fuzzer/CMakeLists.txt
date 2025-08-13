# Find required packages
find_package(Threads REQUIRED)

add_executable(dxc-fuzzer
  MiniHLSLValidator.cpp
)
target_link_libraries(dxc-fuzzer
  dxcompiler            # from DXC build
  clangAST clangBasic   # whatever front-end libs compileToDxil needs
  # LLVMFuzzer            # gives libFuzzer runtime
)
set_target_properties(dxc-fuzzer PROPERTIES
  COMPILE_FLAGS "-fsanitize=address,fuzzer"
  LINK_FLAGS    "-fsanitize=address,fuzzer"
)

# Simple test runner - validates all .hlsl files in a directory
# Commented out - missing test_runner.cpp
# add_executable(test_runner
#   test_runner.cpp
#   MiniHLSLValidator.cpp
# )
# target_link_libraries(test_runner
#   dxcompiler
#   clangAST
#   clangBasic
#   clangFrontend
#   clangLex
#   clangParse
#   clangSema
#   LLVMSupport
#   LLVMOption
#   LLVMCore
# )

# Enable testing
enable_testing()


# MiniHLSL Interpreter - executes MiniHLSL programs and verifies order independence
# Commented out - missing test_interpreter.cpp
# add_executable(minihlsl-interpreter
#   MiniHLSLInterpreter.cpp
#   test_interpreter.cpp
# )

# Standalone MiniHLSL Interpreter - takes HLSL files as command line input
add_executable(minihlsl-standalone
  MiniHLSLInterpreter.cpp
  MiniHLSLValidator.cpp
  minihlsl_interpreter_main.cpp
)
# target_link_libraries(minihlsl-interpreter
#   # Use pthread for std::thread and std::mutex
#   Threads::Threads
#   # Clang/LLVM libraries needed for AST conversion
#   dxcompiler
#   clangAST
#   clangBasic
#   clangFrontend
#   clangLex
#   clangParse
#   clangSema
#   LLVMSupport
#   LLVMOption
#   LLVMCore
# )

target_link_libraries(minihlsl-standalone
  # Use pthread for std::thread and std::mutex
  Threads::Threads
  # Clang/LLVM libraries needed for AST conversion
  dxcompiler
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangParse
  clangSema
  LLVMSupport
  LLVMOption
  LLVMCore
)

# Set C++17 standard for the interpreters (uses std::variant)
# set_target_properties(minihlsl-interpreter PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED ON
# )

set_target_properties(minihlsl-standalone PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Add test that runs all .hlsl files in test_cases directory
# add_test(NAME MiniHLSLValidatorTest COMMAND test_runner test_cases)

# Add test for the interpreter
# add_test(NAME MiniHLSLInterpreterTest COMMAND minihlsl-interpreter)

# MiniHLSL Interpreter Fuzzer - trace-guided fuzzing for interpreter
add_executable(minihlsl-fuzzer
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp  # For loadSeedCorpus
  HLSLProgramGenerator.cpp  # For random program generation when FUZZ_GENERATE_RANDOM=1
  HLSLProgramGeneratorUtils.cpp
  HLSLParticipantPatterns.cpp
  HLSLMutationTracker.cpp  # Needed by HLSLProgramGenerator
  IncrementalFuzzingPipeline.cpp  # For automated incremental fuzzing
  # HLSLSemanticMutator.cpp  # Not needed - mutations are in MiniHLSLInterpreterFuzzer.cpp
)
target_link_libraries(minihlsl-fuzzer
  # Use pthread for std::thread and std::mutex
  Threads::Threads
  # Clang/LLVM libraries needed for AST conversion
  dxcompiler
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangParse
  clangSema
  LLVMSupport
  LLVMOption
  LLVMCore
  # LibFuzzer for fuzzing runtime
  # LLVMFuzzer  # Comment out until linking is fixed
)

# Set C++17 standard
set_target_properties(minihlsl-fuzzer PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)



# Fuzzer for real HLSL files
add_executable(fuzz-hlsl-file
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp
  fuzz_hlsl_file.cpp
)


target_link_libraries(fuzz-hlsl-file
  Threads::Threads
  dxcompiler
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangParse
  clangSema
  LLVMSupport
  LLVMOption
  LLVMCore
)
set_target_properties(fuzz-hlsl-file PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# minihlsl-fuzzer supports two modes:
# 1. Traditional seed corpus mode (default): ./bin/minihlsl-fuzzer seeds -runs=1000
# 2. Random program generation mode: FUZZ_GENERATE_RANDOM=1 ./bin/minihlsl-fuzzer -runs=1000

