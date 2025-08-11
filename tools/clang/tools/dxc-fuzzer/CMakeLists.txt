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
  # MiniHLSLInterpreterFuzzerMutations.cpp  # Commented out - mutations are in MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp  # Added to support loadSeedCorpus
  HLSLProgramGenerator.cpp  # Added for random program generation
  HLSLProgramGeneratorUtils.cpp
  HLSLParticipantPatterns.cpp
  HLSLMutationTracker.cpp
  HLSLSemanticMutator.cpp
  # minihlsl_fuzzer_main.cpp  # Remove custom main - libFuzzer provides main
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

# Set C++17 standard and fuzzing flags
set_target_properties(minihlsl-fuzzer PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "-fsanitize=address,fuzzer -fno-omit-frame-pointer"
  LINK_FLAGS    "-fsanitize=address,fuzzer"
)



# Fuzzer for real HLSL files
add_executable(fuzz-hlsl-file
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterFuzzerMutations.cpp
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

# Incremental HLSL fuzzer with control flow generation
add_executable(fuzz-hlsl-incremental
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzerMutations.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp
  HLSLProgramGenerator.cpp
  HLSLProgramGeneratorUtils.cpp
  HLSLParticipantPatterns.cpp
  HLSLMutationTracker.cpp
  HLSLSemanticMutator.cpp
  fuzz_hlsl_incremental.cpp
)

target_link_libraries(fuzz-hlsl-incremental
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

# Set fuzzing flags for the incremental fuzzer
set_target_properties(fuzz-hlsl-incremental PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "-fsanitize=address,fuzzer -fno-omit-frame-pointer"
  LINK_FLAGS    "-fsanitize=address,fuzzer"
)

