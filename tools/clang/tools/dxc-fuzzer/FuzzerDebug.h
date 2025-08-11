#pragma once

// Debug output control for HLSL fuzzer
// Set FUZZER_DEBUG_OUTPUT=1 at compile time to enable debug output

#ifndef FUZZER_DEBUG_OUTPUT
  #ifdef NDEBUG
    #define FUZZER_DEBUG_OUTPUT 0
  #else
    #define FUZZER_DEBUG_OUTPUT 1
  #endif
#endif

#if FUZZER_DEBUG_OUTPUT
  #include <iostream>
  #define FUZZER_DEBUG_LOG(msg) std::cout << msg
  #define FUZZER_DEBUG_STREAM std::cout
#else
  // No-op for release builds
  #define FUZZER_DEBUG_LOG(msg) ((void)0)
  // Provide a null stream that discards output
  struct NullStream {
    template<typename T>
    NullStream& operator<<(const T&) { return *this; }
  };
  static NullStream nullStream;
  #define FUZZER_DEBUG_STREAM nullStream
#endif