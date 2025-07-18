#include "dxc/dxcapi.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/microcom.h"
#include "dxc/Support/d3dx12.h"
#include "MiniHLSLValidator.h"
#include <windows.h>
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

using Microsoft::WRL::ComPtr;

class DxContext {
public:
  DxContext() { init(); }

  [[nodiscard]] bool valid() const noexcept { return m_device != nullptr; }

  /// Wait until all submitted GPU work is finished.
  void flush() {
    if (!valid()) return;
    m_queue->Signal(m_fence.Get(), ++m_fenceVal);
    if (m_fence->GetCompletedValue() < m_fenceVal) {
      m_fence->SetEventOnCompletion(m_fenceVal, m_fenceEvent);
      ::WaitForSingleObject(m_fenceEvent, INFINITE);
    }
  }

  /// Reset command allocator/list so the caller can record new work.
  void begin() {
    m_allocator->Reset();
    m_cmdList->Reset(m_allocator.Get(), nullptr);
  }

  /// Close and submit the list, leaving context ready for `flush()`.
  void end() {
    m_cmdList->Close();
    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_queue->ExecuteCommandLists(1, lists);
  }

  // Lightweight getters (add more as needed)
  ID3D12Device*                   device()   const { return m_device.Get(); }
  ID3D12GraphicsCommandList*      cmdList()  const { return m_cmdList.Get(); }
  D3D12_GPU_DESCRIPTOR_HANDLE     uavHandle()const { return m_uavHandle; }

private:
  //--------------------------------------------------------------------
  // One-time setup helpers
  //--------------------------------------------------------------------
  void init() {
    createDeviceAndQueue();
    createCommandObjects();
    createFence();
    createBuffers();
    createDescriptorHeap();
  }

  void createDeviceAndQueue() {
    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter> warp;
    ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));

    ThrowIfFailed(D3D12CreateDevice(
        warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));

    D3D12_COMMAND_QUEUE_DESC qdesc = {};
    ThrowIfFailed(m_device->CreateCommandQueue(
        &qdesc, IID_PPV_ARGS(&m_queue)));
  }

  void createCommandObjects() {
    ThrowIfFailed(m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator)));

    ThrowIfFailed(m_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_allocator.Get(), nullptr, IID_PPV_ARGS(&m_cmdList)));
  }

  void createFence() {
    ThrowIfFailed(m_device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceEvent = ::CreateEvent(nullptr, FALSE, FALSE, nullptr);
  }

  void createBuffers() {
    constexpr UINT kBufBytes = 4 * 1024;

    auto defaultProps  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto readbackProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufDesc       = CD3DX12_RESOURCE_DESC::Buffer(
                            kBufBytes,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // GPU UAV buffer
    ThrowIfFailed(m_device->CreateCommittedResource(
        &defaultProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&m_gpuBuf)));

    // CPU-readable buffer
    ThrowIfFailed(m_device->CreateCommittedResource(
        &readbackProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_readbackBuf)));
  }

  void createDescriptorHeap() {
    D3D12_DESCRIPTOR_HEAP_DESC hdesc = {};
    hdesc.NumDescriptors = 1;
    hdesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hdesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    ThrowIfFailed(m_device->CreateDescriptorHeap(
        &hdesc, IID_PPV_ARGS(&m_heapUav)));

    m_uavHandle = m_heapUav->GetGPUDescriptorHandleForHeapStart();

    // Create the UAV
    m_device->CreateUnorderedAccessView(
        m_gpuBuf.Get(), nullptr, nullptr,
        m_heapUav->GetCPUDescriptorHandleForHeapStart());
  }

  // Simple HRESULT wrapper
  static void ThrowIfFailed(HRESULT hr) {
    if (FAILED(hr)) throw std::runtime_error("DxContext init failed");
  }

  //--------------------------------------------------------------------
  // Data members (all private, `m_`-prefixed)
  //--------------------------------------------------------------------
  ComPtr<ID3D12Device>              m_device;
  ComPtr<ID3D12CommandQueue>        m_queue;
  ComPtr<ID3D12CommandAllocator>    m_allocator;
  ComPtr<ID3D12GraphicsCommandList> m_cmdList;
  ComPtr<ID3D12Fence>               m_fence;
  HANDLE                            m_fenceEvent{};
  UINT64                            m_fenceVal = 0;

  // 4-KB UAV + read-back buffer
  ComPtr<ID3D12Resource>            m_gpuBuf;
  ComPtr<ID3D12Resource>            m_readbackBuf;

  // Descriptors
  ComPtr<ID3D12DescriptorHeap>      m_heapUav;
  D3D12_GPU_DESCRIPTOR_HANDLE       m_uavHandle{};

  // (root-sig / PSO caches can be added here later)
};


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

  // First, validate the original shader with MiniHLSL
  minihlsl::MiniHLSLValidator validator;
  auto validationResult = validator.validateSource(std::string(original));
  
  // If original doesn't pass MiniHLSL validation, try to generate compliant variants
  if (!validationResult.isValid) {
    // Generate order-independent variants using miniHLSL generator
    auto orderIndependentVariants = minihlsl::generateOrderIndependentVariants(std::string(original));
    
    // Validate each generated variant
    std::vector<str> validVariants;
    for (const auto& variant : orderIndependentVariants) {
      auto variantValidation = validator.validateSource(variant);
      if (variantValidation.isValid) {
        validVariants.push_back(variant);
      }
    }
    
    if (!validVariants.empty()) {
      return validVariants;
    }
    
    // If no valid variants found, fall back to original method but filter results
  }

  MutationBuilder mb{original};
  if (!mb.valid()) return std::nullopt;

  std::mt19937 rng{std::random_device{}()};

  // Generate mutations using MiniHLSL-compliant patterns only
  auto mutations = std::move(mb)
      // MiniHLSL compliant: Uniform branching only
      .push_front("\n    // 1) MiniHLSL: Uniform condition with wave operations\n"
                  "    if (WaveGetLaneCount() >= 4) {\n"
                  "      float uniformValue = WaveActiveSum(1.0f);\n"
                  "    }\n")
      // MiniHLSL compliant: Order-independent arithmetic
      .push_front("\n    // 2) MiniHLSL: Order-independent lane operations\n"
                  "    uint lane = WaveGetLaneIndex();\n"
                  "    float commutativeResult = float(lane) + float(lane * 2);\n"
                  "    float sum = WaveActiveSum(commutativeResult);\n")
      // MiniHLSL compliant: Associative reductions
      .push_front("\n    // 3) MiniHLSL: Associative wave operations\n"
                  "    uint idx = WaveGetLaneIndex();\n"
                  "    uint product = WaveActiveProduct(idx + 1);\n"
                  "    uint maxVal = WaveActiveMax(idx);\n")
      // MiniHLSL compliant: Deterministic bit operations
      .push_front("\n    // 4) MiniHLSL: Order-independent bit counting\n"
                  "    bool condition = (WaveGetLaneIndex() & 1) == 0;\n"
                  "    uint evenCount = WaveActiveCountBits(condition);\n")
      // MiniHLSL compliant: Multiple uniform conditions
      .push_front("\n    // 5) MiniHLSL: Multiple uniform wave queries\n"
                  "    if (WaveActiveAllEqual(42)) {\n"
                  "      if (WaveActiveAnyTrue(true)) {\n"
                  "        float result = WaveActiveMin(float(WaveGetLaneIndex()));\n"
                  "      }\n"
                  "    }\n")
      .freeze();

  // Filter mutations to ensure MiniHLSL compliance
  std::vector<str> validMutations;
  for (const auto& mutation : mutations) {
    auto mutationValidation = validator.validateSource(mutation);
    if (mutationValidation.isValid) {
      validMutations.push_back(mutation);
    }
  }
  
  return validMutations.empty() ? std::nullopt : std::make_optional(validMutations);
}




// Helper function to compile with specific flags
static ComPtr<IDxcResult> compileWithFlags(
    IDxcCompiler3* pCompiler,
    const DxcBuffer& sourceBuffer,
    const std::vector<LPCWSTR>& args) {
  
  ComPtr<IDxcResult> pResult;
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
  
  ComPtr<IDxcBlob> pShader;
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
  
  ComPtr<IDxcBlobUtf8> pDisasm;
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
    ComPtr<IDxcCompiler3> pCompiler;
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

    // MINIHLSL VALIDATION: Ensure order-independence before testing
    minihlsl::MiniHLSLValidator validator;
    auto validationResult = validator.validateSource(hlslSource);
    
    // If input violates MiniHLSL constraints, try to generate compliant version
    if (!validationResult.isValid) {
      auto compliantVariants = minihlsl::generateOrderIndependentVariants(hlslSource);
      
      bool foundCompliant = false;
      for (const auto& variant : compliantVariants) {
        auto variantValidation = validator.validateSource(variant);
        if (variantValidation.isValid) {
          hlslSource = variant;  // Use compliant variant
          foundCompliant = true;
          break;
        }
      }
      
      // Skip inputs that cannot be made order-independent
      if (!foundCompliant) {
        return 0;
      }
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
  
  // MINIHLSL ENHANCED MUTATION: Generate order-independent mutations
  minihlsl::MiniHLSLValidator validator;
  auto validationResult = validator.validateSource(input);
  
  std::vector<std::string> candidates;
  
  // Generate MiniHLSL-compliant variants
  auto orderIndependentVariants = minihlsl::generateOrderIndependentVariants(input);
  for (const auto& variant : orderIndependentVariants) {
    auto variantValidation = validator.validateSource(variant);
    if (variantValidation.isValid && variant.size() <= MaxSize) {
      candidates.push_back(variant);
    }
  }
  
  // Also try traditional mutations but filter for compliance
  auto traditionalMutations = generate_semantic_mutations(input).has_value()
      ? generate_semantic_mutations(input).value()
      : std::vector<std::string>();
      
  for (const auto& mutation : traditionalMutations) {
    auto mutationValidation = validator.validateSource(mutation);
    if (mutationValidation.isValid && mutation.size() <= MaxSize) {
      candidates.push_back(mutation);
    }
  }
  
  if (candidates.empty()) {
    return 0;
  }
  
  // Select a random MiniHLSL-compliant mutation
  const std::string& selected = candidates[rand() % candidates.size()];
  
  if (selected.size() <= MaxSize) {
    memcpy(Data, selected.c_str(), selected.size());
    return selected.size();
  }
  
  return 0; // Mutation too large
}