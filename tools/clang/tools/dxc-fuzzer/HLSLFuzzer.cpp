//===-- HLSLSemanticFuzzer.cpp - Enhanced Wave Intrinsic Semantic Fuzzer --===//
//
// Enhanced fuzzer for finding semantic bugs in HLSL wave intrinsics
// Focuses on control flow, reconvergence, and correctness rather than crashes
//
//===----------------------------------------------------------------------===//

#include "dxc/dxcapi.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/microcom.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace dxc;

// Global DXC support instance
static DxcDllSupport g_dxcSupport;
static bool g_initialized = false;

// Initialize DXC once
static void InitializeDXC() {
  if (!g_initialized) {
    g_dxcSupport.Initialize();
    g_initialized = true;
  }
}

// wave intrinsics list
static const std::vector<std::string> g_waveIntrinsics = {
  "WaveActiveAllEqual", "WaveActiveAnyTrue", "WaveActiveAllTrue",
  "WaveActiveBallot", "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst",
  "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
  "WaveActiveCountBits", "WavePrefixSum", "WavePrefixProduct",
  "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
  "WavePrefixCountBits", "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor",
  "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor"
};

// Helper function to check if shader contains wave intrinsics
static bool containsWaveIntrinsics(const std::string& hlslSource) {
  for (const auto& intrinsic : g_waveIntrinsics) {
    if (hlslSource.find(intrinsic) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// Generate maximal reconvergence test mutations for wave intrinsics
static std::vector<std::string> generateSemanticMutations(const std::string& originalSource) {
  std::vector<std::string> mutations;
  
  // Find safe insertion point: after opening brace of main function
  size_t mainPos = originalSource.find("void main(");
  if (mainPos == std::string::npos) {
    return mutations; // No main function found
  }
  
  size_t bracePos = originalSource.find("{", mainPos);
  if (bracePos == std::string::npos) {
    return mutations; // No opening brace found
  }
  
  // Safe insertion point: right after the opening brace
  size_t insertionPoint = bracePos + 1;
  
  // Verify we have wave intrinsics in the source
  bool hasWaveIntrinsics = false;
  for (const auto& intrinsic : g_waveIntrinsics) {
    if (originalSource.find(intrinsic) != std::string::npos) {
      hasWaveIntrinsics = true;
      break;
    }
  }
  
  if (!hasWaveIntrinsics) {
    return mutations; // No wave intrinsics found
  }
  
  // Mutation 1: Simple diverge-converge pattern (should reconverge immediately)
  std::string mutation1 = originalSource;
  std::string simpleReconverge = 
    "\n    // Simple maximal reconvergence test\n"
    "    if (WaveGetLaneIndex() % 2 == 0) {\n"
    "        int dummy1 = 1;\n"
    "    } else {\n"
    "        int dummy2 = 2;\n"
    "    }\n"
    "    // Should reconverge here for wave operations\n";
  mutation1.insert(insertionPoint, simpleReconverge);
  mutations.push_back(mutation1);
  
  // Mutation 2: Nested control flow reconvergence
  std::string mutation2 = originalSource;
  std::string nestedReconverge = 
    "\n    // Nested reconvergence test\n"
    "    if (WaveGetLaneIndex() < 16) {\n"
    "        if (WaveGetLaneIndex() % 2 == 0) {\n"
    "            int evenLow = 1;\n"
    "        } else {\n"
    "            int oddLow = 2;\n"
    "        }\n"
    "    } else {\n"
    "        if (WaveGetLaneIndex() % 2 == 0) {\n"
    "            int evenHigh = 3;\n"
    "        } else {\n"
    "            int oddHigh = 4;\n"
    "        }\n"
    "    }\n"
    "    // Maximal reconvergence: all lanes should be active here\n";
  mutation2.insert(insertionPoint, nestedReconverge);
  mutations.push_back(mutation2);
  
  // Mutation 3: Switch-based reconvergence (similar to WebGPU CTS)
  std::string mutation3 = originalSource;
  std::string switchReconverge = 
    "\n    // Switch reconvergence test\n"
    "    switch (WaveGetLaneIndex() % 4) {\n"
    "        case 0: { int lane0 = 10; break; }\n"
    "        case 1: { int lane1 = 20; break; }\n"
    "        case 2: { int lane2 = 30; break; }\n"
    "        case 3: { int lane3 = 40; break; }\n"
    "    }\n"
    "    // Switch exit: should reconverge for wave ops\n";
  mutation3.insert(insertionPoint, switchReconverge);
  mutations.push_back(mutation3);
  
  // Mutation 4: Loop-based reconvergence patterns
  std::string mutation4 = originalSource;
  std::string loopReconverge = 
    "\n    // Loop reconvergence test\n"
    "    for (int i = 0; i < (WaveGetLaneIndex() % 3 + 1); i++) {\n"
    "        int loopVar = i * 10;\n"
    "    }\n"
    "    // Post-loop: maximal reconvergence expected\n";
  mutation4.insert(insertionPoint, loopReconverge);
  mutations.push_back(mutation4);
  
  // Mutation 5: Complex diverge-reconverge with function calls
  std::string mutation5 = originalSource;
  std::string complexReconverge = 
    "\n    // Complex reconvergence with uniform values\n"
    "    float uniformValue = 42.0f;\n"
    "    if (WaveGetLaneIndex() < 8) {\n"
    "        uniformValue *= 2.0f;\n"
    "        if (WaveGetLaneIndex() % 2 == 0) {\n"
    "            uniformValue += 1.0f;\n"
    "        }\n"
    "    } else if (WaveGetLaneIndex() < 16) {\n"
    "        uniformValue *= 3.0f;\n"
    "    } else {\n"
    "        for (int j = 0; j < 2; j++) {\n"
    "            uniformValue += 5.0f;\n"
    "        }\n"
    "    }\n"
    "    // Maximal reconvergence: uniformValue should be consistent for wave ops\n";
  mutation5.insert(insertionPoint, complexReconverge);
  mutations.push_back(mutation5);
  
  // Mutation 6: Conditional execution patterns
  std::string mutation6 = originalSource;
  std::string conditionalTest = 
    "\n    // Conditional execution reconvergence test\n"
    "    bool shouldExecute = (WaveGetLaneIndex() % 8 != 0);\n"
    "    if (shouldExecute) {\n"
    "        float tempValue = 1.0f;\n"
    "        float waveResult = WaveActiveSum(tempValue);\n"
    "    }\n"
    "    // Post-conditional: reconvergence for subsequent wave ops\n";
  mutation6.insert(insertionPoint, conditionalTest);
  mutations.push_back(mutation6);
  
  // Mutation 7: Uniform vs non-uniform reconvergence
  std::string mutation7 = originalSource;
  std::string uniformTest = 
    "\n    // Uniform vs non-uniform reconvergence\n"
    "    bool uniformCondition = true; // Same for all lanes\n"
    "    bool laneCondition = (WaveGetLaneIndex() % 2 == 0); // Per-lane\n"
    "    \n"
    "    if (uniformCondition) {\n"
    "        if (laneCondition) {\n"
    "            int evenValue = 100;\n"
    "        } else {\n"
    "            int oddValue = 200;\n"
    "        }\n"
    "    }\n"
    "    // Uniform condition exit: should have maximal reconvergence\n";
  mutation7.insert(insertionPoint, uniformTest);
  mutations.push_back(mutation7);
  
  return mutations;
}

// Helper function to compile with specific flags
static CComPtr<IDxcResult> compileWithFlags(
    IDxcCompiler3* pCompiler,
    const DxcBuffer& sourceBuffer,
    const std::vector<LPCWSTR>& args) {
  
  CComPtr<IDxcResult> pResult;
  HRESULT hr = pCompiler->Compile(
      &sourceBuffer,
      const_cast<LPCWSTR*>(args.data()),
      static_cast<UINT32>(args.size()),
      nullptr,
      IID_PPV_ARGS(&pResult)
  );
  
  return SUCCEEDED(hr) ? pResult : nullptr;
}

// Helper function to extract compiled bytecode
static std::vector<uint8_t> extractBytecode(IDxcResult* pResult) {
  if (!pResult) return {};
  
  HRESULT status;
  pResult->GetStatus(&status);
  if (FAILED(status)) return {};
  
  CComPtr<IDxcBlob> pShader;
  HRESULT hr = pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pShader), nullptr);
  if (FAILED(hr) || !pShader) return {};
  
  const uint8_t* data = static_cast<const uint8_t*>(pShader->GetBufferPointer());
  size_t size = pShader->GetBufferSize();
  
  return std::vector<uint8_t>(data, data + size);
}

// Helper function to extract disassembly for deeper analysis
static std::string extractDisassembly(IDxcResult* pResult) {
  if (!pResult) return "";
  
  HRESULT status;
  pResult->GetStatus(&status);
  if (FAILED(status)) return "";
  
  CComPtr<IDxcBlobUtf8> pDisasm;
  HRESULT hr = pResult->GetOutput(DXC_OUT_DISASSEMBLY, IID_PPV_ARGS(&pDisasm), nullptr);
  if (FAILED(hr) || !pDisasm) return "";
  
  return std::string(pDisasm->GetStringPointer(), pDisasm->GetStringLength());
}

// Check for wave intrinsic correctness issues, especially reconvergence problems
static bool detectWaveSemanticIssues(const std::string& disasm1, const std::string& disasm2) {
  // Look for suspicious patterns that might indicate semantic bugs
  
  // Check 1: Wave operations should produce similar instruction patterns
  std::vector<std::string> waveOps = {"wave", "ballot", "reduce", "scan", "broadcast"};
  
  for (const auto& op : waveOps) {
    size_t count1 = 0, count2 = 0;
    size_t pos = 0;
    
    // Count occurrences in first disassembly
    while ((pos = disasm1.find(op, pos)) != std::string::npos) {
      count1++;
      pos += op.length();
    }
    
    // Count occurrences in second disassembly
    pos = 0;
    while ((pos = disasm2.find(op, pos)) != std::string::npos) {
      count2++;
      pos += op.length();
    }
    
    // If wave op counts differ significantly, flag as potential issue
    if (count1 > 0 && count2 > 0) {
      float ratio = static_cast<float>(count1) / count2;
      if (ratio < 0.5 || ratio > 2.0) {
        return true; // Significant difference in wave operations
      }
    }
  }
  
  // Check 2: Reconvergence-specific patterns
  std::vector<std::string> reconvergencePatterns = {
    "convergent", "reconverge", "uniform", "divergent", "ballot", "exec"
  };
  
  for (const auto& pattern : reconvergencePatterns) {
    size_t count1 = 0, count2 = 0;
    size_t pos = 0;
    
    while ((pos = disasm1.find(pattern, pos)) != std::string::npos) {
      count1++;
      pos += pattern.length();
    }
    
    pos = 0;
    while ((pos = disasm2.find(pattern, pos)) != std::string::npos) {
      count2++;
      pos += pattern.length();
    }
    
    // Reconvergence patterns should be consistent between optimization levels
    if (count1 != count2 && (count1 > 0 || count2 > 0)) {
      return true; // Reconvergence handling differs between optimizations
    }
  }
  
  // Check 3: Control flow instruction consistency
  std::vector<std::string> controlFlowOps = {"br", "switch", "loop", "if", "else"};
  
  int totalControlFlow1 = 0, totalControlFlow2 = 0;
  for (const auto& cfOp : controlFlowOps) {
    size_t pos = 0;
    while ((pos = disasm1.find(cfOp, pos)) != std::string::npos) {
      totalControlFlow1++;
      pos += cfOp.length();
    }
    
    pos = 0;
    while ((pos = disasm2.find(cfOp, pos)) != std::string::npos) {
      totalControlFlow2++;
      pos += cfOp.length();
    }
  }
  
  // Control flow structure should be preserved for reconvergence
  if (totalControlFlow1 > 0 && totalControlFlow2 > 0) {
    float cfRatio = static_cast<float>(totalControlFlow1) / totalControlFlow2;
    if (cfRatio < 0.7 || cfRatio > 1.4) {
      return true; // Control flow structure changed significantly
    }
  }
  
  return false;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // Skip empty inputs
  if (Size == 0) {
    return 0;
  }

  InitializeDXC();

  try {
    // Create compiler instance
    CComPtr<IDxcCompiler3> pCompiler;
    HRESULT hr = g_dxcSupport.CreateInstance(CLSID_DxcCompiler, &pCompiler);
    if (FAILED(hr)) {
      return 0;
    }

    // Convert input to string
    std::string hlslSource(reinterpret_cast<const char*>(Data), Size);

    // Skip non-wave intrinsic shaders for focused testing
    if (!containsWaveIntrinsics(hlslSource)) {
      return 0;
    }

    // Generate semantic-preserving mutations for differential testing
    std::vector<std::string> mutations = generateSemanticMutations(hlslSource);
    mutations.insert(mutations.begin(), hlslSource); // Include original
    
    // Test each mutation for semantic correctness
    std::vector<LPCWSTR> profiles = {L"cs_6_7", L"ps_6_7"}; // Focus on compute and pixel shaders
    
    std::vector<uint8_t> originalBytecode;
    
    for (size_t mutIdx = 0; mutIdx < mutations.size(); ++mutIdx) {
      const auto& currentSource = mutations[mutIdx];
      
      // Create source buffer for current mutation
      DxcBuffer sourceBuffer;
      sourceBuffer.Ptr = currentSource.c_str();
      sourceBuffer.Size = currentSource.size();
      sourceBuffer.Encoding = CP_UTF8;
      
      for (LPCWSTR profile : profiles) {
        // Enhanced differential testing with semantic validation
        std::vector<LPCWSTR> baseArgs = {
            L"/T", profile,
            L"/E", L"main",
            L"/Vd"  // Disable validation for faster fuzzing
        };

        // Configuration 1: No optimization
        std::vector<LPCWSTR> argsO0 = baseArgs;
        argsO0.push_back(L"/Od");

        // Configuration 2: Full optimization
        std::vector<LPCWSTR> argsO3 = baseArgs;
        argsO3.push_back(L"/O3");

        // Configuration 3: With detailed disassembly
        std::vector<LPCWSTR> argsDisasm = baseArgs;
        argsDisasm.push_back(L"/Od");

        // Compile with shader model 6.7 configurations only
        auto resultO0 = compileWithFlags(pCompiler, sourceBuffer, argsO0);
        auto resultO3 = compileWithFlags(pCompiler, sourceBuffer, argsO3);
        auto resultDisasm = compileWithFlags(pCompiler, sourceBuffer, argsDisasm);

        // Extract bytecode and disassembly for comparison
        auto bytecodeO0 = extractBytecode(resultO0);
        auto bytecodeO3 = extractBytecode(resultO3);
        auto disasmO0 = extractDisassembly(resultDisasm);

        // Check for inconsistencies in compilation success/failure
        bool o0Success = !bytecodeO0.empty();
        bool o3Success = !bytecodeO3.empty();

        // SEMANTIC BUG DETECTION
        
        // Check 1: Optimization level should not affect wave intrinsic compilation success
        if (o0Success != o3Success) {
          // Potential bug: optimization level affects wave intrinsic compilation
          __builtin_trap();
        }
        
        // Check 2: Semantic-preserving mutations should compile identically
        if (mutIdx == 0) {
          originalBytecode = bytecodeO0; // Store original for comparison
        } else if (!originalBytecode.empty() && o0Success) {
          // Compare semantic-preserving mutations
          if (bytecodeO0.size() > 0) {
            float consistency = static_cast<float>(bytecodeO0.size()) / originalBytecode.size();
            
            // Reconvergence mutations should produce similar bytecode sizes
            // Allowing more variance for complex control flow patterns
            if (consistency < 0.5 || consistency > 2.0) {
              // BUG: Reconvergence mutation produced drastically different code
              __builtin_trap();
            }
          }
        }
        
        // Check 2b: Reconvergence-specific validation
        if (mutIdx > 0 && o0Success && !disasmO0.empty()) {
          // Check if reconvergence patterns are being handled correctly
          if (disasmO0.find("wave") != std::string::npos) {
            // Wave operations should be present and correctly handled
            // after reconvergence patterns
            size_t waveCount = 0;
            size_t pos = 0;
            while ((pos = disasmO0.find("wave", pos)) != std::string::npos) {
              waveCount++;
              pos += 4;
            }
            
            // If we have control flow but no wave operations in disasm, 
            // might indicate reconvergence compilation issues
            if (waveCount == 0 && 
                (disasmO0.find("br") != std::string::npos || 
                 disasmO0.find("switch") != std::string::npos)) {
              // BUG: Control flow present but wave operations missing/optimized away incorrectly
              __builtin_trap();
            }
          }
        }

        // Check 3: Wave intrinsic semantic analysis using disassembly
        if (o0Success && o3Success && !disasmO0.empty()) {
          // Extract disassembly from O3 version for comparison
          std::vector<LPCWSTR> argsDisasmO3 = baseArgs;
          argsDisasmO3.push_back(L"/O3");
          auto resultDisasmO3 = compileWithFlags(pCompiler, sourceBuffer, argsDisasmO3);
          auto disasmO3 = extractDisassembly(resultDisasmO3);
          
          if (!disasmO3.empty() && detectWaveSemanticIssues(disasmO0, disasmO3)) {
            // BUG: Wave operations differ unexpectedly between optimization levels
            __builtin_trap();
          }
        }
        
        // Removed cross-version check - focusing only on 6.7

        // If we found a working profile, break (avoid redundant testing)
        if (o0Success || o3Success) {
          break;
        }
      }
    }

  } catch (...) {
    // Catch any exceptions to prevent fuzzer from crashing
    return 0;
  }

  return 0;
}

// Custom mutator for semantic-preserving wave intrinsic mutations
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
  // Only mutate if input contains wave intrinsics
  std::string input(reinterpret_cast<char*>(Data), Size);
  if (!containsWaveIntrinsics(input)) {
    return 0; // Let libFuzzer handle non-wave code
  }
  
  srand(Seed);
  std::vector<std::string> mutations = generateSemanticMutations(input);
  
  if (mutations.empty()) {
    return 0;
  }
  
  // Select a random mutation
  const std::string& selected = mutations[rand() % mutations.size()];
  
  if (selected.size() <= MaxSize) {
    memcpy(Data, selected.c_str(), selected.size());
    return selected.size();
  }
  
  return 0; // Mutation too large
}