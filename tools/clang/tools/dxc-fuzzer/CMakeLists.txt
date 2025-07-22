add_executable(dxc-fuzzer
  HLSLFuzzer.cpp
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
add_executable(test_runner
  test_runner.cpp
  MiniHLSLValidator.cpp
)
target_link_libraries(test_runner
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

# Enable testing
enable_testing()

# MiniHLSL-specific fuzzer - generates only valid MiniHLSL mutations
add_executable(minihlsl-fuzzer
  MiniHLSLFuzzer.cpp
  MiniHLSLValidator.cpp
)
target_link_libraries(minihlsl-fuzzer
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
set_target_properties(minihlsl-fuzzer PROPERTIES
  COMPILE_FLAGS "-fsanitize=address,fuzzer"
  LINK_FLAGS    "-fsanitize=address,fuzzer"
)

# Add test that runs all .hlsl files in test_cases directory
add_test(NAME MiniHLSLValidatorTest COMMAND test_runner test_cases)