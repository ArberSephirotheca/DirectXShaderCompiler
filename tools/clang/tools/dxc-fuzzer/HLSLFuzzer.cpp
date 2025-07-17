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
#include <array>
#include <string_view>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <ranges>
#include <optional>
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

// // wave intrinsics list
// static const std::vector<std::string> g_waveIntrinsics = {
//   "WaveActiveAllEqual", "WaveActiveAnyTrue", "WaveActiveAllTrue",
//   "WaveActiveBallot", "WaveReadLaneAt", "WaveReadFirstLane", "WaveReadLaneFirst",
//   "WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax",
//   "WaveActiveCountBits", "WavePrefixSum", "WavePrefixProduct",
//   "WaveGetLaneIndex", "WaveGetLaneCount", "WaveIsFirstLane",
//   "WavePrefixCountBits", "WaveActiveAnd", "WaveActiveOr", "WaveActiveXor",
//   "WavePrefixAnd", "WavePrefixOr", "WavePrefixXor"
// };

// // Helper function to check if shader contains wave intrinsics
// static bool containsWaveIntrinsics(const std::string& hlslSource) {
//   for (const auto& intrinsic : g_waveIntrinsics) {
//     if (hlslSource.find(intrinsic) != std::string::npos) {
//       return true;
//     }
//   }
//   return false;
// }

// // Helper: locate the opening and matching closing brace of `void main(...)`
// static std::pair<size_t, size_t> findMainBody(const std::string &src) {
//   size_t mainPos  = src.find("void main(");
//   if (mainPos == std::string::npos) return {std::string::npos, std::string::npos};

//   size_t openBrace = src.find('{', mainPos);
//   if (openBrace == std::string::npos) return {std::string::npos, std::string::npos};

//   int depth = 1;
//   for (size_t i = openBrace + 1; i < src.size(); ++i) {
//     if (src[i] == '{') ++depth;
//     else if (src[i] == '}') {
//       --depth;
//       if (depth == 0) return {openBrace, i};
//     }
//   }
//   return {std::string::npos, std::string::npos}; // unmatched
// }

// // Generate maximal reconvergence test mutations for wave intrinsics
// static std::vector<std::string> generateSemanticMutations(const std::string& originalSource) {
//   std::vector<std::string> mutations;
  
//   auto [openBrace, closeBrace] = findMainBody(originalSource);
//   if (openBrace == std::string::npos) return mutations;        // no main()
//   // Locate insertion points -------------------------------------------------
//   size_t mainPos  = originalSource.find("void main(");
//   size_t openPos  = (mainPos != std::string::npos) ?
//                     originalSource.find("{", mainPos) : std::string::npos;
//   if (openPos == std::string::npos) return mutations; // no entry point
//   size_t insertStart = openPos + 1;   // right after '{'
//   size_t closePos    = originalSource.rfind('}'); // crude but works for HLSL one‑liner
//   if (closePos == std::string::npos || closePos <= insertStart) return mutations;

//   if (!containsWaveIntrinsics(originalSource)) return mutations; // nothing to fuzz

  
//   // Helper: push mutation at a chosen offset -------------------------------
//   auto pushAt = [&](const char *snippet, size_t offset) {
//     std::string m = originalSource;
//     m.insert(offset, snippet);
//     mutations.push_back(std::move(m));
//   };

//   auto pushFront = [&](const char *s) { pushAt(s, insertStart); };
//   auto pushTail  = [&](const char *s) { pushAt(s, closePos);    };

//   // ---------------- Original 7 patterns -----------------------------------
//   pushFront("\n    // 1) Simple even/odd reconvergence\n"
//        "    if (WaveGetLaneIndex() % 2 == 0) { int d1 = 1; } else { int d2 = 2; }\n");

//   pushFront("\n    // 2) Nested if reconvergence\n"
//        "    if (WaveGetLaneIndex() < 16) {\n"
//        "      if ((WaveGetLaneIndex() & 1) == 0) int e = 1; else int o = 2;\n"
//        "    } else {\n"
//        "      if ((WaveGetLaneIndex() & 1) == 0) int E = 3; else int O = 4;\n"
//        "    }\n");

//   pushFront("\n    // 3) Switch‑case reconvergence\n"
//        "    switch (WaveGetLaneIndex() & 3) {\n"
//        "      case 0: { int a = 10; break; }\n"
//        "      case 1: { int b = 20; break; }\n"
//        "      case 2: { int c = 30; break; }\n"
//        "      case 3: { int d = 40; break; }\n"
//        "    }\n");

//   pushFront("\n    // 4) Data‑dependent loop reconvergence\n"
//        "    for (int i = 0; i < (WaveGetLaneIndex() % 3 + 1); ++i) { int tmp = i; }\n");

//   pushFront("\n    // 5) Complex branched arithmetic\n"
//        "    float v = 42.0f;\n"
//        "    if (WaveGetLaneIndex() < 8) { v *= 2.0f; if ((WaveGetLaneIndex() & 1)==0) v+=1.0f; }\n"
//        "    else if (WaveGetLaneIndex() < 16) { v *= 3.0f; }\n"
//        "    else { for(int j=0;j<2;++j) v += 5.0f; }\n");

//   pushFront("\n    // 6) Conditional execution around WaveActiveSum\n"
//        "    if ((WaveGetLaneIndex() & 7) != 0) { float t=1.0f; float s=WaveActiveSum(t); }\n");

//   pushFront("\n    // 7) Uniform vs lane‑varying branch\n"
//        "    if (true) { if ((WaveGetLaneIndex() & 1)==0) int ev=100; else int od=200; }\n");

//     const char *randSnippet = "    // 8) Random‑offset reconvergence (syntax‑aware)\n"
//       "    uint idx = WaveGetLaneIndex();\n"
//       "    if (idx & 4) { uint r = WaveActiveCountBits(true); }\n";

//     // Collect candidate offsets: newline boundaries only, so we never split
//     // tokens and we stay inside the function body. Avoid the very last line
//     // to keep tail‑closing brace intact.
//     std::vector<size_t> safe;
//     for (size_t pos = insertStart; pos < closePos; ++pos) {
//       if (originalSource[pos] == '\n') safe.push_back(pos + 1); // insert *after* EOL
//     }

//     if (!safe.empty()) {
//       size_t offset = safe[static_cast<size_t>(rand()) % safe.size()];
//       pushAt(randSnippet, offset);
//     }
    
//   return mutations;
// }


using sv  = std::string_view;
using str = std::string;

inline constexpr std::array<sv, 25> kWaveIntrinsics = {
    "WaveActiveAllEqual",  "WaveActiveAnyTrue",  "WaveActiveAllTrue",
    "WaveActiveBallot",    "WaveReadLaneAt",     "WaveReadFirstLane",
    "WaveReadLaneFirst",   "WaveActiveSum",      "WaveActiveProduct",
    "WaveActiveMin",       "WaveActiveMax",      "WaveActiveCountBits",
    "WavePrefixSum",       "WavePrefixProduct",  "WaveGetLaneIndex",
    "WaveGetLaneCount",    "WaveIsFirstLane",    "WavePrefixCountBits",
    "WaveActiveAnd",       "WaveActiveOr",       "WaveActiveXor",
    "WavePrefixAnd",       "WavePrefixOr",       "WavePrefixXor"};

[[nodiscard]] inline bool contains_wave_intrinsics(sv src) {
  return std::any_of(kWaveIntrinsics.begin(), kWaveIntrinsics.end(), [&](sv intr) {
    return src.find(intr) != sv::npos;
  });
}

class MutationBuilder {
public:
  explicit MutationBuilder(sv original)
      : m_src(original) {
    auto main_pos = m_src.find("void main(");
    auto open_pos = (main_pos == sv::npos) ? sv::npos : m_src.find('{', main_pos);
    m_insert_start = (open_pos == sv::npos) ? sv::npos : open_pos + 1;
    m_close_pos = m_src.rfind('}');
    if (m_close_pos <= m_insert_start) { m_insert_start = sv::npos; }
  }

  [[nodiscard]] bool valid() const { return m_insert_start != sv::npos; }

  MutationBuilder &push_front(sv snippet) {
    return push_at(snippet, m_insert_start);
  }
  MutationBuilder &push_tail(sv snippet) {
    return push_at(snippet, m_close_pos);
  }
  MutationBuilder &push_random(sv snippet, std::mt19937 &rng) {
    std::vector<std::size_t> offsets;
    for (std::size_t pos = m_insert_start; pos < m_close_pos; ++pos) {
      if (m_src[pos] == '\n') offsets.emplace_back(pos + 1);
    }
    if (!offsets.empty()) {
      std::uniform_int_distribution<std::size_t> dist(0, offsets.size() - 1);
      return push_at(snippet, offsets[dist(rng)]);
    }
    return *this; // no safe point – ignore
  }

  [[nodiscard]] std::vector<str> freeze() & { return std::move(m_out); }

private:
  MutationBuilder &push_at(sv snippet, std::size_t offset) {
    if (offset == sv::npos) return *this;
    str mutated(m_src);
    mutated.insert(offset, snippet);
    m_out.emplace_back(std::move(mutated));
    return *this;
  }

  sv                 m_src;
  std::size_t        m_insert_start{sv::npos};
  std::size_t        m_close_pos{sv::npos};
  std::vector<str>   m_out;
};


[[nodiscard]] inline std::optional<std::vector<str>> generate_semantic_mutations(sv original) {
  if (!contains_wave_intrinsics(original)) return std::nullopt; 

  MutationBuilder mb{original};
  if (!mb.valid()) return std::nullopt;

  std::mt19937 rng{std::random_device{}()};

  return std::move(mb)
      .push_front("\n    // 1) Simple even/odd reconvergence\n"
                  "    if (WaveGetLaneIndex() % 2 == 0) { int d1 = 1; } else { int d2 = 2; }\n")
      .push_front("\n    // 2) Nested if reconvergence\n"
                  "    if (WaveGetLaneIndex() < 16) {\n"
                  "      if ((WaveGetLaneIndex() & 1) == 0) int e = 1; else int o = 2;\n"
                  "    } else {\n"
                  "      if ((WaveGetLaneIndex() & 1) == 0) int E = 3; else int O = 4;\n"
                  "    }\n")
      .push_front("\n    // 3) Switch‑case reconvergence\n"
                  "    switch (WaveGetLaneIndex() & 3) {\n"
                  "      case 0: { int a = 10; break; }\n"
                  "      case 1: { int b = 20; break; }\n"
                  "      case 2: { int c = 30; break; }\n"
                  "      case 3: { int d = 40; break; }\n"
                  "    }\n")
      .push_front("\n    // 4) Data‑dependent loop reconvergence\n"
                  "    for (int i = 0; i < (WaveGetLaneIndex() % 3 + 1); ++i) { int tmp = i; }\n")
      .push_front("\n    // 5) Complex branched arithmetic\n"
                  "    float v = 42.0f;\n"
                  "    if (WaveGetLaneIndex() < 8) { v *= 2.0f; if ((WaveGetLaneIndex() & 1)==0) v+=1.0f; }\n"
                  "    else if (WaveGetLaneIndex() < 16) { v *= 3.0f; }\n"
                  "    else { for(int j=0;j<2;++j) v += 5.0f; }\n")
      .push_front("\n    // 6) Conditional execution around WaveActiveSum\n"
                  "    if ((WaveGetLaneIndex() & 7) != 0) { float t=1.0f; float s=WaveActiveSum(t); }\n")
      .push_front("\n    // 7) Uniform vs lane‑varying branch\n"
                  "    if (true) { if ((WaveGetLaneIndex() & 1)==0) int ev=100; else int od=200; }\n")
      .push_front("\n    // 8) Inline function wrapper\n"
                  "    { int Inlined(int x) { return WaveActiveSum(x); } int res = Inlined(1); }\n")
      .push_random("\n    // 9) Random‑offset reconvergence (syntax‑aware)\n"
                   "    uint idx = WaveGetLaneIndex();\n"
                   "    if (idx & 4) { uint r = WaveActiveCountBits(true); }\n", rng)
      .freeze();
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
    "convergent", "uniform"
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
    if (!contains_wave_intrinsics(hlslSource)) {
      return 0;
    }

    // Generate semantic-preserving mutations for differential testing
    std::vector<std::string> mutations = generate_semantic_mutations(hlslSource).has_value()
        ? generate_semantic_mutations(hlslSource).value()
        : std::vector<std::string>();
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
  if (!contains_wave_intrinsics(input)) {
    return 0; // Let libFuzzer handle non-wave code
  }
  
  srand(Seed);
  std::vector<std::string> mutations = generate_semantic_mutations(input).has_value()
      ? generate_semantic_mutations(input).value()
      : std::vector<std::string>();
  
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