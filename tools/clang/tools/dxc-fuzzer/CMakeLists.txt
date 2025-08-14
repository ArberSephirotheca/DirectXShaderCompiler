# =================== Centralized fuzzer/ASan setup ===================
set(THREADS_PREFER_PTHREAD_FLAG ON)  # adds -pthread when needed (Linux/MinGW)
find_package(Threads REQUIRED)

# Toggle once for your whole build
option(ENABLE_FUZZ_ASAN "Enable libFuzzer + AddressSanitizer on fuzz targets" ON)
option(DISABLE_MSVC_STL_ANNOTATIONS "Work around MSVC STL annotate_string mismatch" ON)

# Enable GNU dialect for Clang/GCC so your ({ ... }) macro compiles (optional)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(CMAKE_CXX_EXTENSIONS ON)  # gnu++17 if CXX_STANDARD is 17
endif()

function(enable_fuzz_sanitizers tgt)
  if (NOT ENABLE_FUZZ_ASAN)
    return()
  endif()

  # Only Clang supports -fsanitize=fuzzer
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${tgt} PRIVATE -fsanitize=address,fuzzer)
    target_link_options(${tgt}    PRIVATE -fsanitize=address,fuzzer)
  endif()

  # If linking against prebuilt MSVC libs (dxcompiler/Clang), avoid LNK2038
  # when mixing sanitized and unsanitized code.
  if (DISABLE_MSVC_STL_ANNOTATIONS AND MSVC)
    target_compile_definitions(${tgt} PRIVATE
      _DISABLE_STRING_ANNOTATION
      _DISABLE_VECTOR_ANNOTATION
    )
  endif()
endfunction()

# Keep MSVC iterator debug level consistent across configs
function(sync_msvc_iterator_debug_level tgt)
  if (MSVC)
    target_compile_definitions(${tgt} PRIVATE
      $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>
      $<$<NOT:$<CONFIG:Debug>>:_ITERATOR_DEBUG_LEVEL=0>
    )
  endif()
endfunction()
# =====================================================================

add_executable(dxc-fuzzer
  MiniHLSLValidator.cpp
)

add_executable(minihlsl-standalone
  MiniHLSLInterpreter.cpp
  MiniHLSLValidator.cpp
  minihlsl_interpreter_main.cpp
)
set_target_properties(minihlsl-standalone PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

add_executable(minihlsl-fuzzer
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp
  HLSLProgramGenerator.cpp
  HLSLProgramGeneratorUtils.cpp
  HLSLParticipantPatterns.cpp
  HLSLMutationTracker.cpp
  IncrementalFuzzingPipeline.cpp
)
set_target_properties(minihlsl-fuzzer PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Enable exception handling on Windows
if(MSVC)
  target_compile_options(minihlsl-fuzzer PRIVATE /EHsc)
endif()

add_executable(fuzz-hlsl-file
  MiniHLSLInterpreter.cpp
  MiniHLSLInterpreterFuzzer.cpp
  MiniHLSLInterpreterTraceCapture.cpp
  MiniHLSLValidator.cpp
  fuzz_hlsl_file.cpp
)
set_target_properties(fuzz-hlsl-file PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_link_libraries(dxc-fuzzer
  dxcompiler
  clangAST clangBasic
)

target_link_libraries(minihlsl-standalone
  Threads::Threads
  dxcompiler
  clangAST clangBasic clangFrontend clangLex clangParse clangSema
  LLVMSupport LLVMOption LLVMCore
)

target_link_libraries(minihlsl-fuzzer
  Threads::Threads
  dxcompiler
  clangAST clangBasic clangFrontend clangLex clangParse clangSema
  LLVMSupport LLVMOption LLVMCore
)

target_link_libraries(fuzz-hlsl-file
  Threads::Threads
  dxcompiler
  clangAST clangBasic clangFrontend clangLex clangParse clangSema
  LLVMSupport LLVMOption LLVMCore
)

# --- Apply centralized flags/defs ---
enable_fuzz_sanitizers(dxc-fuzzer)
# Don't use fuzzer mode for minihlsl-fuzzer since it has its own main()
# enable_fuzz_sanitizers(minihlsl-fuzzer)
# enable_fuzz_sanitizers(fuzz-hlsl-file)  # uncomment if you want ASan+Fuzzer here too

# sync_msvc_iterator_debug_level(dxc-fuzzer)
# sync_msvc_iterator_debug_level(minihlsl-fuzzer)
# sync_msvc_iterator_debug_level(fuzz-hlsl-file)
# sync_msvc_iterator_debug_level(minihlsl-standalone)
