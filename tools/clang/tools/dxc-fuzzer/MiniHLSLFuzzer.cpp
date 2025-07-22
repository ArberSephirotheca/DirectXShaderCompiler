#include "MiniHLSLValidator.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/microcom.h"
#include "dxc/dxcapi.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace dxc;
using namespace minihlsl;
// On Linux, DXC uses CComPtr instead of Microsoft::WRL::ComPtr
template <typename T> using ComPtr = CComPtr<T>;

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

// MiniHLSL-compliant wave intrinsics (order-independent)
static constexpr std::array<const char *, 11> kMiniHLSLWaveOps = {
    "WaveActiveSum",     "WaveActiveProduct",   "WaveActiveMax",
    "WaveActiveMin",     "WaveActiveCountBits", "WaveActiveAnyTrue",
    "WaveActiveAllTrue", "WaveActiveAllEqual",  "WaveGetLaneIndex",
    "WaveGetLaneCount",  "WaveIsFirstLane"};

// Forbidden operations in MiniHLSL (order-dependent)
static constexpr std::array<const char *, 8> kForbiddenWaveOps = {
    "WavePrefixSum",  "WavePrefixProduct", "WavePrefixCountBits",
    "WavePrefixAnd",  "WavePrefixOr",      "WavePrefixXor",
    "WaveReadLaneAt", "WaveReadLaneFirst"};

// Check if source contains wave intrinsics
bool contains_wave_intrinsics(const std::string &source) {
  for (const auto &op : kMiniHLSLWaveOps) {
    if (source.find(op) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// Check if source contains forbidden operations
bool contains_forbidden_operations(const std::string &source) {
  for (const auto &op : kForbiddenWaveOps) {
    if (source.find(op) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// MiniHLSL mutation generator class
class MiniHLSLMutationGenerator {
public:
  explicit MiniHLSLMutationGenerator(const std::string &original,
                                     std::mt19937 &rng)
      : original_(original), rng_(rng) {
    find_main_function();
  }

  // Generate MiniHLSL-compliant mutations
  std::vector<std::string> generate_mutations() {
    std::vector<std::string> mutations;

    if (!is_valid_for_mutation()) {
      return mutations;
    }

    // Generate various types of order-independent mutations
    mutations.push_back(add_deterministic_condition());
    mutations.push_back(add_commutative_arithmetic());
    mutations.push_back(add_associative_wave_ops());
    mutations.push_back(add_uniform_branching());
    mutations.push_back(add_lane_index_operations());
    mutations.push_back(duplicate_wave_operations());
    mutations.push_back(reorder_commutative_operations());

    // Filter out invalid mutations
    std::vector<std::string> valid_mutations;
    MiniHLSLValidator validator;

    for (const auto &mutation : mutations) {
      if (!mutation.empty()) {
        auto result = validator.validate_source(mutation);
        if (result.is_ok()) {
          valid_mutations.push_back(mutation);
        }
      }
    }

    return valid_mutations;
  }

private:
  std::string original_;
  std::mt19937 &rng_;
  size_t main_start_ = std::string::npos;
  size_t main_body_start_ = std::string::npos;
  size_t main_body_end_ = std::string::npos;

  void find_main_function() {
    // Find main function (or main1, main2, etc.)
    size_t pos = original_.find("void main");
    if (pos != std::string::npos) {
      main_start_ = pos;

      // Find opening brace
      size_t brace_pos = original_.find('{', pos);
      if (brace_pos != std::string::npos) {
        main_body_start_ = brace_pos + 1;

        // Find matching closing brace
        int brace_count = 1;
        size_t search_pos = brace_pos + 1;
        while (search_pos < original_.size() && brace_count > 0) {
          if (original_[search_pos] == '{')
            brace_count++;
          else if (original_[search_pos] == '}')
            brace_count--;
          search_pos++;
        }
        if (brace_count == 0) {
          main_body_end_ = search_pos - 1;
        }
      }
    }
  }

  bool is_valid_for_mutation() {
    return main_start_ != std::string::npos &&
           main_body_start_ != std::string::npos &&
           main_body_end_ != std::string::npos &&
           !contains_forbidden_operations(original_);
  }

  std::string insert_at_random_location(const std::string &code) {
    if (main_body_start_ >= main_body_end_)
      return "";

    // Find newlines for good insertion points
    std::vector<size_t> newlines;
    for (size_t i = main_body_start_; i < main_body_end_; ++i) {
      if (original_[i] == '\n') {
        newlines.push_back(i + 1);
      }
    }

    if (newlines.empty()) {
      newlines.push_back(main_body_start_);
    }

    std::uniform_int_distribution<size_t> dist(0, newlines.size() - 1);
    size_t insert_pos = newlines[dist(rng_)];

    std::string result = original_;
    result.insert(insert_pos, code);
    return result;
  }

  // Mutation 1: Add deterministic conditional statements
  std::string add_deterministic_condition() {
    std::vector<std::string> conditions = {
        "\n    // MiniHLSL: Uniform condition\n"
        "    if (WaveGetLaneCount() >= 32) {\n"
        "        float uniform_value = WaveActiveSum(1.0f);\n"
        "    }\n",

        "\n    // MiniHLSL: Lane index condition (deterministic)\n"
        "    if ((WaveGetLaneIndex() & 1) == 0) {\n"
        "        float even_result = float(WaveGetLaneIndex());\n"
        "    }\n",

        "\n    // MiniHLSL: Deterministic wave query\n"
        "    if (WaveActiveAllEqual(42)) {\n"
        "        bool any_active = WaveActiveAnyTrue(true);\n"
        "    }\n"};

    std::uniform_int_distribution<size_t> dist(0, conditions.size() - 1);
    return insert_at_random_location(conditions[dist(rng_)]);
  }

  // Mutation 2: Add commutative arithmetic operations
  std::string add_commutative_arithmetic() {
    std::vector<std::string> arithmetic = {
        "\n    // MiniHLSL: Commutative arithmetic\n"
        "    uint lane = WaveGetLaneIndex();\n"
        "    float a = float(lane) + float(lane * 2);\n"
        "    float b = a * 3.0f + 1.0f;\n"
        "    float sum = WaveActiveSum(a + b);\n",

        "\n    // MiniHLSL: Order-independent calculations\n"
        "    uint idx = WaveGetLaneIndex();\n"
        "    float x = float(idx * idx + idx + 1);\n"
        "    float max_x = WaveActiveMax(x);\n",

        "\n    // MiniHLSL: Associative operations\n"
        "    uint lane_id = WaveGetLaneIndex();\n"
        "    uint product = WaveActiveProduct(lane_id + 1);\n"
        "    uint bit_and = WaveActiveAnd(lane_id);\n"};

    std::uniform_int_distribution<size_t> dist(0, arithmetic.size() - 1);
    return insert_at_random_location(arithmetic[dist(rng_)]);
  }

  // Mutation 3: Add associative wave operations
  std::string add_associative_wave_ops() {
    std::vector<std::string> wave_ops = {
        "\n    // MiniHLSL: Multiple associative reductions\n"
        "    uint lane = WaveGetLaneIndex();\n"
        "    uint sum_result = WaveActiveSum(lane);\n"
        "    uint min_result = WaveActiveMin(lane);\n"
        "    uint max_result = WaveActiveMax(lane);\n",

        "\n    // MiniHLSL: Bitwise operations (order-independent)\n"
        "    uint idx = WaveGetLaneIndex();\n"
        "    uint or_result = WaveActiveOr(idx);\n"
        "    uint xor_result = WaveActiveXor(idx);\n"
        "    uint and_result = WaveActiveAnd(idx);\n",

        "\n    // MiniHLSL: Count operations\n"
        "    bool condition = (WaveGetLaneIndex() & 3) == 0;\n"
        "    uint count = WaveActiveCountBits(condition);\n"};

    std::uniform_int_distribution<size_t> dist(0, wave_ops.size() - 1);
    return insert_at_random_location(wave_ops[dist(rng_)]);
  }

  // Mutation 4: Add uniform branching patterns
  std::string add_uniform_branching() {
    std::vector<std::string> branches = {
        "\n    // MiniHLSL: Nested uniform conditions\n"
        "    if (WaveIsFirstLane()) {\n"
        "        if (WaveGetLaneCount() > 16) {\n"
        "            float temp = 42.0f;\n"
        "        }\n"
        "    }\n",

        "\n    // MiniHLSL: Wave property branching\n"
        "    if (WaveActiveAnyTrue(true)) {\n"
        "        uint lane_count = WaveGetLaneCount();\n"
        "        if (lane_count == 32) {\n"
        "            float result = WaveActiveSum(float(WaveGetLaneIndex()));\n"
        "        }\n"
        "    }\n"};

    std::uniform_int_distribution<size_t> dist(0, branches.size() - 1);
    return insert_at_random_location(branches[dist(rng_)]);
  }

  // Mutation 5: Add lane index based operations
  std::string add_lane_index_operations() {
    std::vector<std::string> lane_ops = {
        "\n    // MiniHLSL: Lane index based calculations\n"
        "    uint my_lane = WaveGetLaneIndex();\n"
        "    float lane_value = float(my_lane * my_lane);\n"
        "    float total = WaveActiveSum(lane_value);\n",

        "\n    // MiniHLSL: Even/odd lane pattern\n"
        "    bool is_even = (WaveGetLaneIndex() & 1) == 0;\n"
        "    uint even_count = WaveActiveCountBits(is_even);\n"
        "    uint odd_count = WaveGetLaneCount() - even_count;\n"};

    std::uniform_int_distribution<size_t> dist(0, lane_ops.size() - 1);
    return insert_at_random_location(lane_ops[dist(rng_)]);
  }

  // Mutation 6: Duplicate existing wave operations (should be idempotent)
  std::string duplicate_wave_operations() {
    std::string result = original_;

    // Find existing wave operations and duplicate them
    for (const auto &op : kMiniHLSLWaveOps) {
      size_t pos = result.find(op);
      if (pos != std::string::npos) {
        // Find the end of the statement
        size_t end_pos = result.find(';', pos);
        if (end_pos != std::string::npos) {
          std::string statement = result.substr(pos, end_pos - pos + 1);
          std::string duplicate =
              "\n    // MiniHLSL: Duplicated operation\n    float dup_" +
              statement;
          result = insert_at_random_location(duplicate);
          break;
        }
      }
    }

    return result;
  }

  // Mutation 7: Reorder commutative operations
  std::string reorder_commutative_operations() {
    // This is a simplified version - real implementation would parse and
    // reorder AST
    std::string code = "\n    // MiniHLSL: Reordered commutative ops\n"
                       "    uint a = WaveGetLaneIndex();\n"
                       "    uint b = WaveGetLaneCount();\n"
                       "    float result1 = WaveActiveSum(float(a + b));\n"
                       "    float result2 = WaveActiveSum(float(b + a)); // "
                       "Commutative reorder\n";

    return insert_at_random_location(code);
  }
};

// Test one input through MiniHLSL validation and DXC compilation
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0)
    return 0;

  InitializeDXC();

  try {
    std::string hlsl_source(reinterpret_cast<const char *>(Data), Size);

    // Skip inputs that don't contain wave intrinsics
    if (!contains_wave_intrinsics(hlsl_source)) {
      return 0;
    }

    // Skip inputs that contain forbidden operations
    if (contains_forbidden_operations(hlsl_source)) {
      return 0;
    }

    // Validate with MiniHLSLValidator first
    MiniHLSLValidator validator;
    auto validation_result = validator.validate_source(hlsl_source);

    if (!validation_result.is_ok()) {
      return 0; // Skip invalid MiniHLSL
    }

    // Generate mutations and test each one
    std::mt19937 rng(std::hash<std::string>{}(hlsl_source));
    MiniHLSLMutationGenerator generator(hlsl_source, rng);
    auto mutations = generator.generate_mutations();

    // Test original + all mutations
    std::vector<std::string> all_variants = {hlsl_source};
    all_variants.insert(all_variants.end(), mutations.begin(), mutations.end());

    // Try to compile each variant with DXC
    ComPtr<IDxcCompiler3> pCompiler;
    HRESULT hr = g_dxcSupport.CreateInstance(CLSID_DxcCompiler, &pCompiler);
    if (FAILED(hr))
      return 0;

    std::vector<LPCWSTR> args = {L"/T",   L"cs_6_0", L"/E",
                                 L"main", L"/Od",    L"/Vd"};

    bool original_compiled = false;

    for (size_t i = 0; i < all_variants.size(); ++i) {
      const auto &variant = all_variants[i];

      // Create source buffer
      DxcBuffer sourceBuffer;
      sourceBuffer.Ptr = variant.c_str();
      sourceBuffer.Size = variant.size();
      sourceBuffer.Encoding = CP_UTF8;

      // Try to compile
      ComPtr<IDxcResult> pResult;
      hr = pCompiler->Compile(&sourceBuffer, const_cast<LPCWSTR *>(args.data()),
                              static_cast<UINT32>(args.size()), nullptr,
                              IID_PPV_ARGS(&pResult));

      if (SUCCEEDED(hr)) {
        HRESULT compileStatus;
        pResult->GetStatus(&compileStatus);

        bool compiled = SUCCEEDED(compileStatus);

        if (i == 0) {
          original_compiled = compiled;
        } else {
          // All mutations should have same compilation status as original
          if (compiled != original_compiled) {
            // BUG: MiniHLSL mutation changed compilation behavior
            __builtin_trap();
          }
        }
      }
    }

  } catch (...) {
    // Catch any exceptions to prevent fuzzer from crashing
    return 0;
  }

  return 0;
}

// Custom mutator that generates MiniHLSL-compliant mutations
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  if (Size == 0)
    return 0;

  std::string input(reinterpret_cast<char *>(Data), Size);

  // Only mutate if input contains wave intrinsics and is MiniHLSL-valid
  if (!contains_wave_intrinsics(input) ||
      contains_forbidden_operations(input)) {
    return 0;
  }

  // Validate input first
  MiniHLSLValidator validator;
  auto validation_result = validator.validate_source(input);
  if (!validation_result.is_ok()) {
    return 0;
  }

  std::mt19937 rng(Seed);
  MiniHLSLMutationGenerator generator(input, rng);
  auto mutations = generator.generate_mutations();

  if (mutations.empty()) {
    return 0;
  }

  // Select a random mutation
  std::uniform_int_distribution<size_t> dist(0, mutations.size() - 1);
  const auto &selected = mutations[dist(rng)];

  if (selected.size() <= MaxSize) {
    memcpy(Data, selected.c_str(), selected.size());
    return selected.size();
  }

  return 0;
}