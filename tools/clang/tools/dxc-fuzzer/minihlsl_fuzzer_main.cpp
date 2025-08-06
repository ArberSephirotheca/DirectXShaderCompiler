// Main entry point for minihlsl-fuzzer
// This provides a simple main function that calls the libFuzzer entry points

#include <cstdint>
#include <cstddef>
#include <iostream>

// LibFuzzer entry points
extern "C" {
  int LLVMFuzzerInitialize(int* argc, char*** argv);
  int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);
}

int main(int argc, char** argv) {
  // Initialize the fuzzer
  LLVMFuzzerInitialize(&argc, &argv);
  
  // For now, just run a simple test
  std::cout << "MiniHLSL Fuzzer\n";
  std::cout << "This is a placeholder main function.\n";
  std::cout << "To use with libFuzzer, link with -fsanitize=fuzzer\n";
  
  // Test with empty input
  const uint8_t emptyData[] = {0};
  LLVMFuzzerTestOneInput(emptyData, 0);
  
  return 0;
}