add_executable(dxc-fuzzer
  HLSLFuzzer.cpp
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