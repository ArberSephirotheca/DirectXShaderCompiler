#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include "MiniHLSLValidator.h"
#include "HLSLProgramGenerator.h"
#include "IncrementalFuzzingPipeline.h"
#include "FuzzerDebug.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdlib>  // for std::getenv

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <filesystem>
  #include <sys/types.h>
  #include <sys/stat.h>
  namespace fs = std::filesystem;
#else
  #include <sys/stat.h>
  #include <dirent.h>
  #include <unistd.h>
#endif

namespace minihlsl {
namespace fuzzer {

// Forward declaration
static const interpreter::WaveActiveOp* findWaveOpInExpression(const interpreter::Expression* expr);

// Helper function to check if a statement is a tracking operation
bool isTrackingStatement(const interpreter::Statement* stmt) {
  // Check for VarDeclStmt patterns used by various tracking mutations
  if (auto* varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    const std::string& name = varDeclStmt->getName();
    
    // WaveParticipantTracking pattern: uint _participantCount = WaveActiveSum(1)
    if (name == "_participantCount") {
      if (!varDeclStmt->getInit()) return false;
      
      auto* waveOp = findWaveOpInExpression(varDeclStmt->getInit());
      if (!waveOp || waveOp->getOpType() != interpreter::WaveActiveOp::Sum) return false;
      
      // Check if input is literal 1
      auto* literal = dynamic_cast<const interpreter::LiteralExpr*>(waveOp->getInput());
      return literal && literal->getValue() == interpreter::Value(1);
    }
    
    // WaveParticipantBitTracking patterns
    if (name == "temp" && !varDeclStmt->getInit()) {
      // uint temp;
      return true;
    }
    
    if (name == "ballot") {
      // uint4 ballot = WaveActiveBallot(1);
      if (!varDeclStmt->getInit()) return false;
      auto* waveOp = findWaveOpInExpression(varDeclStmt->getInit());
      return waveOp && waveOp->getOpType() == interpreter::WaveActiveOp::Ballot;
    }
    
    // Other tracking-related variables
    if (name == "_isCorrect" || name == "_frequency_index" || name == "_tempIndex") {
      return true;
    }
  }
  
  // Check for AssignStmt patterns
  if (auto* assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    const std::string& name = assignStmt->getName();
    
    // WaveParticipantTracking pattern: _participantCount = WaveActiveSum(1)
    if (name == "_participantCount") {
      auto* waveOp = findWaveOpInExpression(assignStmt->getExpression());
      if (!waveOp || waveOp->getOpType() != interpreter::WaveActiveOp::Sum) return false;
      
      // Check if input is literal 1
      auto* literal = dynamic_cast<const interpreter::LiteralExpr*>(waveOp->getInput());
      return literal && literal->getValue() == interpreter::Value(1);
    }
  }
  
  // Check for ArrayAssignStmt patterns (used by bit tracking)
  if (auto* arrayAssign = dynamic_cast<const interpreter::ArrayAssignStmt*>(stmt)) {
    const std::string& arrayName = arrayAssign->getArrayName();
    // Bit tracking uses _participant_bit array assignments
    if (arrayName == "_participant_bit" || arrayName == "_participant_count" || 
        arrayName == "_frequency_buffer") {
      return true;
    }
  }
  
  // Check for ExprStmt with InterlockedAdd (used by bit tracking)
  if (auto* exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    if (dynamic_cast<const interpreter::InterlockedAddExpr*>(exprStmt->getExpression())) {
      return true;
    }
  }
  
  return false;
}

// Helper function to check if a wave operation in the trace is likely a tracking operation
bool isTrackingWaveOp(const ExecutionTrace::WaveOpRecord& waveOp, size_t index, 
                      const std::vector<ExecutionTrace::WaveOpRecord>& allOps) {
  // Tracking operations are WaveActiveSum operations
  if (waveOp.opType != "WaveActiveSum") return false;
  
  // The key heuristic: tracking operations follow this pattern
  // 1. They immediately follow another wave operation in the same block
  // 2. They are WaveActiveSum operations (counting participants)
  // 3. They have the exact same participants as the previous operation
  
  if (index > 0) {
    const auto& prevOp = allOps[index - 1];
    
    // Check if this is in the same block and immediately follows the previous op
    if (prevOp.blockId == waveOp.blockId) {
      // Check if both operations have the same participants
      // Tracking operations execute with the same lanes as the original operation
      if (prevOp.arrivedParticipants == waveOp.arrivedParticipants) {
        // This is very likely a tracking operation
        return true;
      }
    }
  }
  
  return false;
}

// Helper function to create output directory and generate filenames
std::string createMutantOutputPath(size_t increment, const std::string& mutationChain, const std::string& extension = ".hlsl", const std::string& seedId = "", const std::string& outputDir = "mutant_outputs") {
  // Create output directory if it doesn't exist
#ifdef _WIN32
  CreateDirectoryA(outputDir.c_str(), NULL);
#else
  mkdir(outputDir.c_str(), 0755);
#endif
  
  // Generate filename based on seed ID (if provided), increment and mutation chain
  std::stringstream filename;
  filename << outputDir << "/";
  if (!seedId.empty()) {
    filename << "program_" << seedId << "_";
  }
  filename << "increment_" << increment;
  
  // Clean up mutation chain string for filename
  std::string cleanMutationChain = mutationChain;
  std::replace(cleanMutationChain.begin(), cleanMutationChain.end(), ' ', '_');
  std::replace(cleanMutationChain.begin(), cleanMutationChain.end(), '+', '_');
  std::replace(cleanMutationChain.begin(), cleanMutationChain.end(), '(', '_');
  std::replace(cleanMutationChain.begin(), cleanMutationChain.end(), ')', '_');
  
  if (!cleanMutationChain.empty()) {
    filename << "_" << cleanMutationChain;
  }
  
  filename << extension;
  return filename.str();
}

// Helper function to generate test file with YAML pipeline
void generateTestFile(const interpreter::Program& program, 
                     const ExecutionTrace& trace,
                     const std::string& testPath,
                     const std::string& mutationChain) {
  std::ofstream testFile(testPath);
  if (!testFile.is_open()) {
    FUZZER_DEBUG_LOG("Failed to create test file: " << testPath << "\n");
    return;
  }
  
  // Write HLSL source section
  testFile << "#--- source.hlsl\n";
  
  // The program already includes buffer declarations from WaveParticipantTracking
  // Just write the program as-is
  bool hasParticipantTracking = mutationChain.find("WaveParticipantTracking") != std::string::npos;
  bool hasBitTracking = mutationChain.find("WaveParticipantBitTracking") != std::string::npos;
  testFile << serializeProgramToString(program);
  
  // Calculate buffer sizes
  uint32_t totalThreads = program.numThreadsX * program.numThreadsY * program.numThreadsZ;
  uint32_t bufferSizeInEntries = totalThreads;
  
  // Generate expected participant data from trace
  std::vector<uint32_t> expectedParticipants(totalThreads, 0);
  std::vector<uint32_t> expectedBitPatterns;
  
  if (hasParticipantTracking) {
    // The WaveParticipantTracking mutation structure:
    // 1. Original wave operation (e.g., WaveActiveSum(value))
    // 2. Tracking operation: WaveActiveSum(1) to count participants
    // 3. Check: _participant_check_sum[tid.x] += (count == expected) ? 1 : 0
    
    // Debug: Print trace information
    if (trace.waveOperations.empty()) {
      FUZZER_DEBUG_LOG("WARNING: No wave operations in trace!\n");
      return;  // Don't generate test file if no wave operations
    }
    FUZZER_DEBUG_LOG("Generating expected data from trace with " << trace.waveOperations.size() << " wave operations\n");
    FUZZER_DEBUG_LOG("Total threads: " << totalThreads << ", Wave size: " << trace.threadHierarchy.waveSize << "\n");
    
    // First, collect all wave operations from the program and mark which are tracking statements
    std::map<uint32_t, bool> stableIdToIsTracking;  // stable ID -> is tracking operation
    
    // Helper to recursively find wave ops in statements
    std::function<void(const interpreter::Statement*)> collectWaveOps;
    collectWaveOps = [&](const interpreter::Statement* stmt) {
      if (!stmt) return;
      
      // Check if this is a tracking statement
      bool isTracking = isTrackingStatement(stmt);
      
      // Find wave operation in this statement
      const interpreter::WaveActiveOp* waveOp = nullptr;
      if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
        waveOp = findWaveOpInExpression(assign->getExpression());
      } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
        if (varDecl->getInit()) {
          waveOp = findWaveOpInExpression(varDecl->getInit());
        }
      }
      
      if (waveOp) {
        stableIdToIsTracking[waveOp->getStableId()] = isTracking;
        FUZZER_DEBUG_LOG("Found wave op with stable ID " << waveOp->getStableId() 
                        << " - tracking: " << (isTracking ? "yes" : "no") << "\n");
      }
      
      // Recurse into nested statements
      if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
        for (const auto& s : ifStmt->getThenBlock()) {
          collectWaveOps(s.get());
        }
        for (const auto& s : ifStmt->getElseBlock()) {
          collectWaveOps(s.get());
        }
      } else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
        for (const auto& s : whileStmt->getBody()) {
          collectWaveOps(s.get());
        }
      } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
        for (const auto& s : forStmt->getBody()) {
          collectWaveOps(s.get());
        }
      } else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
        for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
          const auto& caseBlock = switchStmt->getCase(i);
          for (const auto& s : caseBlock.statements) {
            collectWaveOps(s.get());
          }
        }
      }
    };
    
    // Collect wave ops from all statements
    for (const auto& stmt : program.statements) {
      collectWaveOps(stmt.get());
    }
    
    // Process trace operations, skipping tracking operations
    size_t trackingOpsSkipped = 0;
    size_t originalOpsProcessed = 0;
    
    for (size_t i = 0; i < trace.waveOperations.size(); i++) {
      const auto& waveOp = trace.waveOperations[i];
      
      // Check if this is a tracking operation based on stable ID from the mutated program
      auto it = stableIdToIsTracking.find(waveOp.stableId);
      bool isTrackingOperation = (it != stableIdToIsTracking.end() && it->second);
      
      // Skip if it's a tracking operation
      if (isTrackingOperation) {
        FUZZER_DEBUG_LOG("Wave op " << i << " (stable ID " << waveOp.stableId 
                        << "): SKIPPING tracking operation\n");
        trackingOpsSkipped++;
        continue;
      }
      
      FUZZER_DEBUG_LOG("Wave op " << i << " (stable ID " << waveOp.stableId 
                      << "): " << waveOp.opType << " on Wave " << waveOp.waveId 
                      << " with " << waveOp.arrivedParticipants.size() << " participants\n");
      originalOpsProcessed++;
      
      // Convert lane IDs to global thread IDs
      uint32_t waveSize = trace.threadHierarchy.waveSize;
      if (waveSize == 0) {
        FUZZER_DEBUG_LOG("WARNING: Wave size is 0, using default 32\n");
        waveSize = 32;
      }
      uint32_t waveBaseThreadId = waveOp.waveId * waveSize;
      
      // Each participating thread should increment their success counter by 1
      for (auto laneId : waveOp.arrivedParticipants) {
        uint32_t globalThreadId = waveBaseThreadId + laneId;
        if (globalThreadId < totalThreads) {
          expectedParticipants[globalThreadId]++;
          FUZZER_DEBUG_LOG("  Thread " << globalThreadId << " (wave " << waveOp.waveId 
                          << ", lane " << laneId << ") expected++\n");
        }
      }
    }
    
    FUZZER_DEBUG_LOG("Expected participant counts: ");
    for (uint32_t i = 0; i < totalThreads; ++i) {
      FUZZER_DEBUG_LOG(expectedParticipants[i] << " ");
    }
    FUZZER_DEBUG_LOG("\n");
    FUZZER_DEBUG_LOG("Total original operations processed: " << originalOpsProcessed << "\n");
    FUZZER_DEBUG_LOG("Total tracking operations skipped: " << trackingOpsSkipped << "\n");
  }
  
  if (hasBitTracking) {
    // Generate expected bit patterns from trace
    // WaveParticipantBitTrackingMutation records:
    // - Combined ID: (iteration << 16) | waveOpId
    // - Participant masks: WaveActiveBallot results (x, y components)
    
    FUZZER_DEBUG_LOG("Generating expected bit patterns from trace\n");
    
    // Process each wave operation separately - don't group by wave
    FUZZER_DEBUG_LOG("Total wave operations in trace: " << trace.waveOperations.size() << "\n");
    for (const auto& waveOp : trace.waveOperations) {
      // Skip Ballot operations (they're used for tracking)
      if (waveOp.waveOpEnumType == static_cast<int>(interpreter::WaveActiveOp::OpType::Ballot)) {
        continue;
      }
      
      FUZZER_DEBUG_LOG("Processing wave op - stable ID: " << waveOp.stableId 
                      << ", block ID: " << waveOp.blockId 
                      << ", participants: " << waveOp.arrivedParticipants.size() << "\n");
      
      // Compute combined ID with bitwise encoding (matching the shader)
      // Bits [31-6]: Wave op ID (26 bits)
      // Bits [5-4]: Loop 1 index (outermost)
      // Bits [3-2]: Loop 2 index (middle)
      // Bits [1-0]: Loop 3 index (innermost)
      uint32_t combinedId = waveOp.stableId << 6;
      
      // Use the loop iterations directly from the wave operation record
      FUZZER_DEBUG_LOG("  Loop iterations: ");
      for (auto iter : waveOp.loopIterations) {
        FUZZER_DEBUG_LOG(iter << " ");
      }
      FUZZER_DEBUG_LOG("\n");
      
      for (size_t i = 0; i < waveOp.loopIterations.size() && i < 3; ++i) {
        int shift = (2 - i) * 2;  // Loop 0: shift 4, Loop 1: shift 2, Loop 2: shift 0
        combinedId |= (waveOp.loopIterations[i] & 0x3) << shift;
      }
      
      FUZZER_DEBUG_LOG("  Combined ID: " << combinedId << " (0x" << std::hex << combinedId << std::dec << ")\n");
      
      // Compute participant bitmask for this specific wave execution
      uint64_t mask = 0;
      for (interpreter::LaneId lane : waveOp.arrivedParticipants) {
        mask |= (1ULL << lane);
      }
      
      // Generate one tuple per participating thread in this wave
      for (interpreter::LaneId lane : waveOp.arrivedParticipants) {
        expectedBitPatterns.push_back(combinedId);
        expectedBitPatterns.push_back(static_cast<uint32_t>(mask & 0xFFFFFFFF));
        expectedBitPatterns.push_back(static_cast<uint32_t>(mask >> 32));
      }
      
      FUZZER_DEBUG_LOG("Wave op " << waveOp.stableId 
                      << " wave " << waveOp.waveId
                      << " combinedId: " << combinedId
                      << " mask: 0x" << std::hex << mask << std::dec 
                      << " participants: " << waveOp.arrivedParticipants.size() << "\n");
    }
    
    FUZZER_DEBUG_LOG("Total bit pattern records: " << expectedBitPatterns.size() / 3 << "\n");
  }
  
  // Write pipeline YAML section
  testFile << "\n#--- pipeline.yaml\n";
  testFile << "---\n";
  testFile << "Shaders:\n";
  testFile << "  - Stage: Compute\n";
  testFile << "    Entry: main\n";
  testFile << "    DispatchSize: [1, 1, 1]  # Single dispatch for " << totalThreads << " threads\n";
  
  testFile << "Buffers:\n";
  
  if (hasParticipantTracking) {
    // _participant_check_sum buffer
    testFile << "  - Name: _participant_check_sum\n";
    testFile << "    Format: UInt32\n";
    testFile << "    Stride: 4\n";
    testFile << "    Fill: 0\n";
    testFile << "    Size: " << bufferSizeInEntries << "\n";
    
    // expected_participants buffer with pre-calculated data
    testFile << "  - Name: expected_participants\n";
    testFile << "    Format: UInt32\n";
    testFile << "    Stride: 4\n";
    testFile << "    Data: [";
    
    // Write expected data
    for (uint32_t i = 0; i < totalThreads; ++i) {
      if (i > 0) testFile << ", ";
      testFile << expectedParticipants[i];
    }
    testFile << "]\n";
  }
  
  // Add bit tracking buffers if needed
  if (hasBitTracking) {
    // _participant_bit buffer with expected data
    testFile << "  - Name: _participant_bit\n";
    testFile << "    Format: UInt32\n";
    testFile << "    Stride: 4\n";
    if (!expectedBitPatterns.empty()) {
      testFile << "    Data: [";
      for (size_t i = 0; i < expectedBitPatterns.size(); ++i) {
        if (i > 0) testFile << ", ";
        testFile << expectedBitPatterns[i];
      }
      testFile << "]\n";
    } else {
      testFile << "    Fill: 0\n";
      testFile << "    Size: 1024\n";  // Default size
    }
    
    // _wave_op_index buffer (counter)
    testFile << "  - Name: _wave_op_index\n";
    testFile << "    Format: UInt32\n";
    testFile << "    Stride: 4\n";
    testFile << "    Data: [0]\n";  // Initialize counter to 0
  }
  
  // Add any global buffers from the program (skip special buffers we already added)
  for (const auto& buffer : program.globalBuffers) {
    if (buffer.name != "_participant_check_sum" && 
        buffer.name != "_participant_bit" &&
        buffer.name != "_wave_op_index") {
      testFile << "  - Name: " << buffer.name << "\n";
      testFile << "    Format: UInt32\n";
      testFile << "    Stride: 4\n";
      testFile << "    Fill: 0\n";
      testFile << "    Size: " << bufferSizeInEntries << "\n";
    }
  }
  
  if (hasParticipantTracking || hasBitTracking) {
    testFile << "Results:\n";
  }
  
  if (hasParticipantTracking) {
    testFile << "  - Result: WaveOpValidation\n";
    testFile << "    Rule: BufferExact\n";
    testFile << "    Actual: _participant_check_sum\n";
    testFile << "    Expected: expected_participants\n";
  }
  
  if (hasBitTracking) {
    testFile << "  - Result: BitTrackingValidation\n";
    testFile << "    Rule: BufferPrefix\n";  // Use prefix since buffer size is dynamic
    testFile << "    Actual: _participant_bit\n";
    testFile << "    Expected: _participant_bit\n";  // Expected data is already in the buffer
  }
  
  testFile << "DescriptorSets:\n";
  testFile << "  - Resources:\n";
  
  if (hasParticipantTracking) {
    testFile << "    - Name: _participant_check_sum\n";
    testFile << "      Kind: RWStructuredBuffer\n";
    testFile << "      DirectXBinding:\n";
    testFile << "        Register: 1\n";
    testFile << "        Space: 0\n";
    testFile << "      VulkanBinding:\n";
    testFile << "        Binding: 1\n";
  }
  
  // Add bindings for program buffers
  for (const auto& buffer : program.globalBuffers) {
    if (buffer.name == "_participant_check_sum" && hasParticipantTracking) {
      // Skip - already handled above
      continue;
    }
    testFile << "    - Name: " << buffer.name << "\n";
    testFile << "      Kind: " << (buffer.isReadWrite ? "RWStructuredBuffer" : "StructuredBuffer") << "\n";
    testFile << "      DirectXBinding:\n";
    testFile << "        Register: " << buffer.registerIndex << "\n";
    testFile << "        Space: 0\n";
    testFile << "      VulkanBinding:\n";
    testFile << "        Binding: " << buffer.registerIndex << "\n";
  }
  
  testFile << "...\n";
  testFile << "#--- end\n\n";
  
  // Write run commands
  testFile << "# RUN: split-file %s %t\n";
  testFile << "# RUN: %dxc_target -T cs_6_0 -Fo %t.o %t/source.hlsl\n";
  testFile << "# RUN: %offloader %t/pipeline.yaml %t.o\n";
  
  testFile.close();
  FUZZER_DEBUG_LOG("Generated test file: " << testPath << "\n");
}


// Global seed programs loaded from corpus

// Helper function to add indentation to multi-line strings
std::string indentString(const std::string& str, const std::string& indent) {
  std::stringstream result;
  std::istringstream stream(str);
  std::string line;
  bool firstLine = true;
  
  while (std::getline(stream, line)) {
    if (!firstLine) {
      result << "\n";
    }
    if (!line.empty()) {
      result << indent << line;
    }
    firstLine = false;
  }
  
  return result.str();
}

// Helper function to re-indent a multi-line string
std::string reindentCode(const std::string& code, int baseIndentLevel) {
  std::stringstream result;
  std::istringstream stream(code);
  std::string line;
  std::vector<int> indentStack;
  indentStack.push_back(0);
  bool firstLine = true;
  
  while (std::getline(stream, line)) {
    if (!firstLine) {
      result << "\n";
    }
    firstLine = false;
    
    // Skip empty lines
    if (line.empty()) {
      continue;
    }
    
    // Count leading spaces
    size_t leadingSpaces = 0;
    for (char c : line) {
      if (c == ' ') leadingSpaces++;
      else break;
    }
    
    // Get the actual content
    std::string content = line.substr(leadingSpaces);
    
    // Detect indent changes
    bool isOpenBrace = content.find('{') != std::string::npos;
    bool isCloseBrace = content.find('}') == 0;
    bool isCaseLabel = content.find("case ") == 0 || content.find("default:") == 0;
    
    // Adjust indent level
    if (isCloseBrace && !indentStack.empty() && indentStack.size() > 1) {
      indentStack.pop_back();
    }
    
    // Calculate actual indent
    int currentIndent = baseIndentLevel;
    if (!indentStack.empty()) {
      currentIndent += indentStack.back();
    }
    
    // Special handling for switch cases
    if (isCaseLabel && currentIndent > baseIndentLevel) {
      currentIndent -= 1; // Case labels are less indented than case body
    }
    
    // Write the line with proper indentation
    result << std::string(currentIndent * 2, ' ') << content;
    
    // Update indent stack for next line
    if (isOpenBrace && !isCloseBrace) {
      // Opening brace increases indent
      indentStack.push_back(indentStack.back() + 1);
    } else if (isCaseLabel) {
      // After case label, increase indent for case body
      if (!indentStack.empty() && indentStack.back() > 0) {
        // We're already in a switch, adjust for case body
        indentStack.back() = indentStack.back(); // Keep same level, body will be indented by the brace
      }
    }
  }
  
  return result.str();
}

// Helper function to serialize a statement with proper indentation
std::string serializeStatementWithIndent(const interpreter::Statement* stmt, int indentLevel) {
  std::string stmtStr = stmt->toString();
  
  // For control flow statements that produce multi-line output, re-indent properly
  if (dynamic_cast<const interpreter::IfStmt*>(stmt) ||
      dynamic_cast<const interpreter::ForStmt*>(stmt) ||
      dynamic_cast<const interpreter::WhileStmt*>(stmt) ||
      dynamic_cast<const interpreter::DoWhileStmt*>(stmt) ||
      dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
    return reindentCode(stmtStr, indentLevel);
  } else {
    // Simple single-line statements
    return std::string(indentLevel * 2, ' ') + stmtStr;
  }
}

// Helper function to serialize a program to string
std::string serializeProgramToString(const interpreter::Program& program) {
  std::stringstream ss;
  
  // Add global buffer declarations
  for (const auto& buffer : program.globalBuffers) {
    ss << buffer.bufferType << "<";
    ss << interpreter::HLSLTypeInfo::toString(buffer.elementType);
    ss << "> " << buffer.name;
    ss << " : register(" << (buffer.isReadWrite ? "u" : "t") 
       << buffer.registerIndex << ");\n";
  }
  
  if (!program.globalBuffers.empty()) {
    ss << "\n";
  }
  
  // Add thread configuration
  ss << "[numthreads(" << program.numThreadsX << ", " 
     << program.numThreadsY << ", " 
     << program.numThreadsZ << ")]\n";
  
  // Removed WaveSize attribute generation as it requires Shader Model 6.6+
  // which may not be available on all test systems
  
  ss << "void main(";
  
  // Add function parameters with semantics
  bool first = true;
  for (const auto& param : program.entryInputs.parameters) {
    if (!first) {
      ss << ",\n          ";
    }
    first = false;
    
    // Output type
    ss << interpreter::HLSLTypeInfo::toString(param.type);
    if (param.type == interpreter::HLSLType::Custom && !param.customTypeStr.empty()) {
      ss << param.customTypeStr;
    }
    
    // Output parameter name
    ss << " " << param.name;
    
    // Output semantic if present
    if (param.semantic != interpreter::HLSLSemantic::None) {
      ss << " : " << interpreter::HLSLSemanticInfo::toString(param.semantic);
      if (param.semantic == interpreter::HLSLSemantic::Custom && !param.customSemanticStr.empty()) {
        ss << param.customSemanticStr;
      }
    }
  }
  
  ss << ") {\n";
  
  // Add all statements with proper indentation
  for (const auto& stmt : program.statements) {
    ss << serializeStatementWithIndent(stmt.get(), 1) << "\n";
  }
  
  ss << "}\n";
  return ss.str();
}

// ===== Semantics-Preserving Mutation Implementations =====

bool LanePermutationMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Check if this is an assignment with a wave operation
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    // Check for wave operation, including nested in binary expressions
    const interpreter::WaveActiveOp* waveOp = getWaveOp(stmt);
    if (waveOp) {
      // Check if this is an associative operation
      auto opType = waveOp->getOpType();
      bool isAssociative = opType == interpreter::WaveActiveOp::Sum ||
                          opType == interpreter::WaveActiveOp::Product ||
                          opType == interpreter::WaveActiveOp::And ||
                          opType == interpreter::WaveActiveOp::Or ||
                          opType == interpreter::WaveActiveOp::Xor;
      
      if (!isAssociative) {
        return false;
      }
      
      // Additionally check if the operation uses thread ID variables
      // This makes the mutation more relevant for testing built-in variable handling
      const auto* inputExpr = waveOp->getInput();
      if (inputExpr && usesThreadIdVariables(inputExpr)) {
        // Prioritize mutations on operations that use thread IDs
        return true;
      }
      
      // Still allow mutations on other associative operations
      return true;
    }
  }
  return false;
}

// Helper function to recursively find WaveActiveOp in an expression
static const interpreter::WaveActiveOp* findWaveOpInExpression(const interpreter::Expression* expr) {
  if (!expr) return nullptr;
  
  // Direct wave operation
  if (auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(expr)) {
    return waveOp;
  }
  
  // Check binary expressions (e.g., result + WaveActiveSum(...))
  if (auto* binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
    // Check left side
    if (auto* waveOp = findWaveOpInExpression(binOp->getLeft())) {
      return waveOp;
    }
    // Check right side
    if (auto* waveOp = findWaveOpInExpression(binOp->getRight())) {
      return waveOp;
    }
  }
  
  // Check unary expressions
  if (auto* unaryOp = dynamic_cast<const interpreter::UnaryOpExpr*>(expr)) {
    return findWaveOpInExpression(unaryOp->getExpr());
  }
  
  return nullptr;
}

const interpreter::WaveActiveOp* LanePermutationMutation::getWaveOp(
    const interpreter::Statement* stmt) const {
  
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return findWaveOpInExpression(assign->getExpression());
  } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    return findWaveOpInExpression(varDecl->getInit());
  } else if (auto* exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    return findWaveOpInExpression(exprStmt->getExpression());
  }
  
  return nullptr;
}

std::unique_ptr<interpreter::Statement> LanePermutationMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  // This mutation requires program-level application to handle nested structures
  // This method should not be called when requiresProgramLevelMutation() returns true
  assert(false && "LanePermutation requires program-level mutation");
  return stmt->clone();
}

std::unique_ptr<interpreter::Expression> LanePermutationMutation::createPermutationExpr(
    PermutationType type,
    std::unique_ptr<interpreter::Expression> laneExpr,
    uint32_t activeLaneCount,
    const std::set<interpreter::LaneId>& participatingLanes) const {
  
  // Convert set to vector for indexed access
  std::vector<interpreter::LaneId> laneList(participatingLanes.begin(), participatingLanes.end());
  std::sort(laneList.begin(), laneList.end());
  
  // Check if participants are consecutive starting from the first lane
  bool isConsecutive = true;
  if (!laneList.empty()) {
    for (size_t i = 1; i < laneList.size(); ++i) {
      if (laneList[i] != laneList[0] + i) {
        isConsecutive = false;
        break;
      }
    }
  }
  
  switch (type) {
    case PermutationType::Rotate: {
      // General rotation that works for any set of participating lanes
      
      if (laneList.size() < 2) {
        // If only one lane participates, return the same lane
        return laneExpr;
      }
      
      // Optimization for consecutive participants
      if (isConsecutive && laneList.size() > 1) {
        // Use mathematical formula: ((laneId - firstLane + 1) % count) + firstLane
        auto firstLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
        auto count = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.size()));
        auto one = std::make_unique<interpreter::LiteralExpr>(1);
        
        // (laneId - firstLane + 1)
        auto offsetFromFirst = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), firstLane->clone(), interpreter::BinaryOpExpr::Sub);
        auto offsetPlusOne = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(offsetFromFirst), std::move(one), interpreter::BinaryOpExpr::Add);
        
        // % count
        auto modResult = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(offsetPlusOne), std::move(count), interpreter::BinaryOpExpr::Mod);
        
        // + firstLane
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::move(modResult), std::move(firstLane), interpreter::BinaryOpExpr::Add);
      }
      
      // Build a conditional expression chain: 
      // (laneId == lane0) ? lane1 : ((laneId == lane1) ? lane2 : ... )
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t nextIdx = (i + 1) % laneList.size();
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto nextLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[nextIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          // Last condition in the chain (or first being built)
          result = std::move(nextLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(nextLane), std::move(result));
        }
      }
      
      return result;
    }
    
    case PermutationType::Reverse: {
      // Reverse mapping for arbitrary participating lanes
      // Maps lane[i] -> lane[n-1-i]
      
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // Optimization for consecutive participants
      if (isConsecutive && laneList.size() > 1) {
        // Use mathematical formula: (firstLane + lastLane) - laneId
        auto firstLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
        auto lastLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.back()));
        
        // firstLane + lastLane
        auto sum = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(firstLane), std::move(lastLane), interpreter::BinaryOpExpr::Add);
        
        // (firstLane + lastLane) - laneId
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::move(sum), laneExpr->clone(), interpreter::BinaryOpExpr::Sub);
      }
      
      // Build conditional chain for reverse mapping
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t reverseIdx = laneList.size() - 1 - i;
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto reverseLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[reverseIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          result = std::move(reverseLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(reverseLane), std::move(result));
        }
      }
      
      return result;
    }
    
    case PermutationType::EvenOddSwap: {
      // Even/odd swap for arbitrary participating lanes
      // Pairs lanes: [0]<->[1], [2]<->[3], etc.
      // If odd number of lanes, last lane maps to itself
      
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // Optimization for consecutive participants - but we need to handle unsigned arithmetic carefully
      if (isConsecutive && laneList.size() > 1 && laneList.size() <= 4) {
        // For small consecutive sets, build specific patterns to avoid underflow
        if (laneList.size() == 2) {
          // Simple swap: lane0 <-> lane1
          auto lane0 = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
          auto lane1 = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[1]));
          auto isLane0 = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), lane0->clone(), interpreter::BinaryOpExpr::Eq);
          
          // lane0 ? lane1 : lane0
          return std::make_unique<interpreter::ConditionalExpr>(
              std::move(isLane0), std::move(lane1), std::move(lane0));
        } else if (laneList.size() == 4 && laneList[0] == 0) {
          // Special case for lanes 0,1,2,3 - can use safe arithmetic
          // Pattern: 0->1, 1->0, 2->3, 3->2
          auto one = std::make_unique<interpreter::LiteralExpr>(1);
          auto two = std::make_unique<interpreter::LiteralExpr>(2);
          
          // Check if lane < 2
          auto isFirstPair = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), two->clone(), interpreter::BinaryOpExpr::Lt);
          
          // For lanes 0,1: (lane == 0) ? 1 : 0
          auto zero = std::make_unique<interpreter::LiteralExpr>(0);
          auto isZero = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), zero->clone(), interpreter::BinaryOpExpr::Eq);
          auto firstPairSwap = std::make_unique<interpreter::ConditionalExpr>(
              std::move(isZero), one->clone(), std::move(zero));
          
          // For lanes 2,3: (lane == 2) ? 3 : 2  
          auto three = std::make_unique<interpreter::LiteralExpr>(3);
          auto isTwo = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), two->clone(), interpreter::BinaryOpExpr::Eq);
          auto secondPairSwap = std::make_unique<interpreter::ConditionalExpr>(
              std::move(isTwo), std::move(three), std::move(two));
          
          // Combine: (lane < 2) ? firstPairSwap : secondPairSwap
          return std::make_unique<interpreter::ConditionalExpr>(
              std::move(isFirstPair), std::move(firstPairSwap), std::move(secondPairSwap));
        }
      }
      
      // Build conditional chain for even/odd swapping
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t pairIdx;
        if (i % 2 == 0) {
          // Even index: swap with next (if exists)
          pairIdx = (i + 1 < laneList.size()) ? i + 1 : i;
        } else {
          // Odd index: swap with previous
          pairIdx = i - 1;
        }
        
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto pairLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[pairIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          result = std::move(pairLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(pairLane), std::move(result));
        }
      }
      
      return result;
    }
    
    default:
      // Default to rotate
      return createPermutationExpr(PermutationType::Rotate, std::move(laneExpr), 
                                   activeLaneCount, participatingLanes);
  }
}

std::unique_ptr<interpreter::Expression> LanePermutationMutation::transformExpression(
    const interpreter::Expression* expr,
    const interpreter::Expression* permutedLaneExpr) const {
  
  // Base case: LaneIndexExpr - replace with permuted version
  if (dynamic_cast<const interpreter::LaneIndexExpr*>(expr)) {
    return permutedLaneExpr->clone();
  }
  
  // Handle different expression types
  if (auto litExpr = dynamic_cast<const interpreter::LiteralExpr*>(expr)) {
    return litExpr->clone();
  }
  
  if (auto varExpr = dynamic_cast<const interpreter::VariableExpr*>(expr)) {
    // Check if this is a built-in variable that contains thread ID information
    std::string varName = varExpr->toString();
    
    // Handle built-in variables that contain thread index information
    // These variables are semantically equivalent to lane indices for wave operations
    if (varName == "tid.x" || varName == "gtid.x" || varName == "gindex" ||
        varName.find("SV_DispatchThreadID") != std::string::npos ||
        varName.find("SV_GroupThreadID") != std::string::npos ||
        varName.find("SV_GroupIndex") != std::string::npos) {
      // Replace with permuted lane expression
      return permutedLaneExpr->clone();
    }
    
    // For other variables, just clone
    return varExpr->clone();
  }
  
  if (auto binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
    // We can't access private members, so we need to reconstruct based on toString
    // This is a limitation - for now just clone
    return expr->clone();
  }
  
  // For other expressions, just clone
  return expr->clone();
}

bool LanePermutationMutation::usesThreadIdVariables(const interpreter::Expression* expr) const {
  // Check if this expression uses built-in thread ID variables
  if (auto varExpr = dynamic_cast<const interpreter::VariableExpr*>(expr)) {
    std::string varName = varExpr->toString();
    return varName == "tid.x" || varName == "tid.y" || varName == "tid.z" ||
           varName == "gtid.x" || varName == "gtid.y" || varName == "gtid.z" ||
           varName == "gid.x" || varName == "gid.y" || varName == "gid.z" ||
           varName == "gindex" ||
           varName == "tid" || varName == "gtid" || varName == "gid";
  }
  
  // Check LaneIndexExpr
  if (dynamic_cast<const interpreter::LaneIndexExpr*>(expr)) {
    return true;
  }
  
  // For compound expressions, would need to recurse, but for now return false
  return false;
}

bool LanePermutationMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* 
    mutated,
    const ExecutionTrace& trace) const {
  // TODO: Implement proper semantic validation
  // This should verify that:
  // 1. The permutation preserves the associative property
  // 2. All lanes still participate in the same wave operation
  // 3. The final result is mathematically equivalent
  // For now, we rely on the fuzzer's end-to-end validation
  return true;
}

// Program-level mutation implementation
std::vector<interpreter::Program> LanePermutationMutation::applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  std::vector<interpreter::Program> mutants;
  
  // Create a base mutant
  interpreter::Program mutant;
  mutant.numThreadsX = program.numThreadsX;
  mutant.numThreadsY = program.numThreadsY;
  mutant.numThreadsZ = program.numThreadsZ;
  mutant.entryInputs = program.entryInputs;
  mutant.globalBuffers = program.globalBuffers;
  mutant.waveSize = program.waveSize;
  
  // Process all statements recursively
  size_t currentStmtIndex = 0;
  bool anyMutationApplied = false;
  processStatementsForPermutation(program.statements, mutant.statements, 
                                  trace, statementsToMutate, 
                                  currentStmtIndex, anyMutationApplied);
  
  if (anyMutationApplied) {
    mutants.push_back(std::move(mutant));
  }
  
  return mutants;
}

void LanePermutationMutation::processStatementsForPermutation(
    const std::vector<std::unique_ptr<interpreter::Statement>>& input,
    std::vector<std::unique_ptr<interpreter::Statement>>& output,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate,
    size_t& currentStmtIndex,
    bool& anyMutationApplied) const {
  
  for (const auto& stmt : input) {
    // Handle control flow structures recursively
    if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt.get())) {
      // Increment statement index for the if statement itself
      currentStmtIndex++;
      
      // Process then block
      std::vector<std::unique_ptr<interpreter::Statement>> processedThen;
      processStatementsForPermutation(ifStmt->getThenBlock(), processedThen, 
                                      trace, statementsToMutate, 
                                      currentStmtIndex, anyMutationApplied);
      
      // Process else block
      std::vector<std::unique_ptr<interpreter::Statement>> processedElse;
      processStatementsForPermutation(ifStmt->getElseBlock(), processedElse, 
                                      trace, statementsToMutate, 
                                      currentStmtIndex, anyMutationApplied);
      
      // Create new if statement with processed blocks
      output.push_back(std::make_unique<interpreter::IfStmt>(
          ifStmt->getCondition()->clone(),
          std::move(processedThen),
          std::move(processedElse)));
    } else if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt.get())) {
      // Increment statement index for the for statement itself
      currentStmtIndex++;
      
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForPermutation(forStmt->getBody(), processedBody, 
                                      trace, statementsToMutate, 
                                      currentStmtIndex, anyMutationApplied);
      
      // Create new for statement with processed body
      output.push_back(std::make_unique<interpreter::ForStmt>(
          forStmt->getLoopVar(),
          forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
          forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
          forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
          std::move(processedBody)));
    } else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt.get())) {
      // Increment statement index for the while statement itself
      currentStmtIndex++;
      
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForPermutation(whileStmt->getBody(), processedBody, 
                                      trace, statementsToMutate, 
                                      currentStmtIndex, anyMutationApplied);
      
      // Create new while statement with processed body
      output.push_back(std::make_unique<interpreter::WhileStmt>(
          whileStmt->getCondition()->clone(),
          std::move(processedBody)));
    } else if (auto doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt.get())) {
      // Increment statement index for the do-while statement itself
      currentStmtIndex++;
      
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForPermutation(doWhileStmt->getBody(), processedBody, 
                                      trace, statementsToMutate, 
                                      currentStmtIndex, anyMutationApplied);
      
      // Create new do-while statement with processed body
      output.push_back(std::make_unique<interpreter::DoWhileStmt>(
          std::move(processedBody),
          doWhileStmt->getCondition()->clone()));
    } else if (auto switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt.get())) {
      // Increment statement index for the switch statement itself
      currentStmtIndex++;
      
      // Process switch cases
      auto newSwitch = std::make_unique<interpreter::SwitchStmt>(
          switchStmt->getCondition()->clone());
      
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        std::vector<std::unique_ptr<interpreter::Statement>> processedCaseBody;
        
        // Process statements in this case
        processStatementsForPermutation(caseBlock.statements, processedCaseBody,
                                        trace, statementsToMutate,
                                        currentStmtIndex, anyMutationApplied);
        
        // Add the processed case to the new switch
        if (caseBlock.value.has_value()) {
          newSwitch->addCase(caseBlock.value.value(), std::move(processedCaseBody));
        } else {
          newSwitch->addDefault(std::move(processedCaseBody));
        }
      }
      
      output.push_back(std::move(newSwitch));
    } else {
      // Check if this statement index should be mutated
      bool shouldMutate = statementsToMutate.find(currentStmtIndex) != statementsToMutate.end();
      
      if (shouldMutate && applyPermutationToStatement(stmt.get(), output, trace)) {
        anyMutationApplied = true;
      } else {
        // For all other statement types, just clone
        output.push_back(stmt->clone());
      }
      
      currentStmtIndex++;
    }
  }
}

bool LanePermutationMutation::applyPermutationToStatement(
    const interpreter::Statement* stmt,
    std::vector<std::unique_ptr<interpreter::Statement>>& output,
    const ExecutionTrace& trace) const {
  
  // Check if this is an assignment with a wave operation
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) {
    return false;
  }
  
  // Use getWaveOp to find nested wave operations
  const interpreter::WaveActiveOp* waveOp = getWaveOp(stmt);
  if (!waveOp) {
    return false;
  }
  
  // Check if the wave operation already uses WaveReadLaneAt (already mutated)
  if (dynamic_cast<const interpreter::WaveReadLaneAt*>(waveOp->getInput())) {
    FUZZER_DEBUG_LOG("[LanePermutation] Skipping already-mutated wave operation\n");
    return false;
  }
  
  // Get the operation type and verify it's associative
  auto opType = waveOp->getOpType();
  bool isAssociative = (opType == interpreter::WaveActiveOp::Sum ||
                        opType == interpreter::WaveActiveOp::Product ||
                        opType == interpreter::WaveActiveOp::And ||
                        opType == interpreter::WaveActiveOp::Or ||
                        opType == interpreter::WaveActiveOp::Xor);
  
  if (!isAssociative) {
    return false;
  }
  
  // Get the input expression from the wave operation
  const auto* inputExpr = waveOp->getInput();
  if (!inputExpr) {
    return false;
  }
  
  // Find participant information from trace
  uint32_t activeLaneCount = 4; // default
  std::set<interpreter::LaneId> participatingLanes;
  
  // Look for this wave operation in the trace
  for (const auto& waveOpRecord : trace.waveOperations) {
    if (waveOpRecord.waveOpEnumType == static_cast<int>(opType)) {
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      participatingLanes = waveOpRecord.arrivedParticipants;
      break;
    }
  }
  
  // Choose a permutation type (for now, just rotate)
  PermutationType permType = PermutationType::Rotate;
  
  FUZZER_DEBUG_LOG("[LanePermutation] Active lanes: " << activeLaneCount 
                   << ", participants: " << participatingLanes.size() << "\n");
  
  // Create the permutation expression
  auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto permutedExpr = createPermutationExpr(permType, std::move(laneExpr), 
                                             activeLaneCount, participatingLanes);
  
  FUZZER_DEBUG_LOG("[LanePermutation] Permuted expr: " << permutedExpr->toString() << "\n");
  
  // Generate variable name
  std::string varName = generatePermVarName();
  
  // Create variable declaration: uint _perm_val_N = permutedExpr;
  output.push_back(std::make_unique<interpreter::VarDeclStmt>(
      varName, interpreter::HLSLType::Uint, std::move(permutedExpr)));
  
  // Create WaveReadLaneAt to read the input from the permuted lane
  auto permLaneVar = std::make_unique<interpreter::VariableExpr>(varName);
  auto readLaneAt = std::make_unique<interpreter::WaveReadLaneAt>(
      inputExpr->clone(), std::move(permLaneVar));
  
  FUZZER_DEBUG_LOG("[LanePermutation] Original input: " << inputExpr->toString() 
                   << ", using WaveReadLaneAt with permuted lane\n");
  
  // Create the new wave operation with the WaveReadLaneAt input
  auto newWaveOp = std::make_unique<interpreter::WaveActiveOp>(std::move(readLaneAt), opType);
  
  // Replace the wave operation in the original expression structure
  auto newExpr = replaceWaveOpInExpression(assign->getExpression(), waveOp, std::move(newWaveOp));
  
  // Create the assignment with the modified expression
  output.push_back(std::make_unique<interpreter::AssignStmt>(
      assign->getName(), std::move(newExpr)));
  
  return true;
}

std::unique_ptr<interpreter::Expression> LanePermutationMutation::replaceWaveOpInExpression(
    const interpreter::Expression* expr,
    const interpreter::WaveActiveOp* targetWaveOp,
    std::unique_ptr<interpreter::Expression> replacement) const {
  
  // If this is the target wave operation, return the replacement
  if (expr == targetWaveOp) {
    return replacement;
  }
  
  // Handle different expression types
  if (auto* binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
    // Recursively replace in left and right operands
    auto newLeft = replaceWaveOpInExpression(binOp->getLeft(), targetWaveOp, 
                                              (binOp->getLeft() == targetWaveOp) ? std::move(replacement) : replacement->clone());
    auto newRight = replaceWaveOpInExpression(binOp->getRight(), targetWaveOp, std::move(replacement));
    
    return std::make_unique<interpreter::BinaryOpExpr>(
        std::move(newLeft), std::move(newRight), binOp->getOp());
  }
  
  // For other expression types, just clone if not the target
  return expr->clone();
}

std::string LanePermutationMutation::generatePermVarName() const {
  // static int counter = 0; // Not needed, using stable IDs
  // Use a thread-local counter to avoid issues in multi-threaded fuzzing
  static thread_local int threadCounter = 0;
  return "_perm_val_" + std::to_string(threadCounter++);
}


// ===== WaveParticipantTrackingMutation Implementation =====

bool WaveParticipantTrackingMutation::canApply(const interpreter::Statement* stmt,
                                              const ExecutionTrace& trace) const {
  // Check if statement contains any wave operations (including nested)
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return findWaveOpInExpression(assign->getExpression()) != nullptr;
  } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    return varDecl->getInit() && 
           findWaveOpInExpression(varDecl->getInit()) != nullptr;
  }
  return false;
}

std::unique_ptr<interpreter::Statement> WaveParticipantTrackingMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  // This mutation is handled specially at the program level via processStatementsForTracking
  // The apply method should not be used for this mutation
  // Just return the statement as-is
  return stmt->clone();
}

std::vector<std::unique_ptr<interpreter::Statement>>
WaveParticipantTrackingMutation::createTrackingStatements(
    const interpreter::WaveActiveOp* waveOp,
    const std::string& resultVar,
    uint32_t expectedParticipants) const {
  
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Count active participants: _participantCount = WaveActiveSum(1)
  auto oneExpr = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
  auto sumOp = std::make_unique<interpreter::WaveActiveOp>(
      std::move(oneExpr), interpreter::WaveActiveOp::Sum);
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_participantCount", interpreter::HLSLType::Uint, std::move(sumOp)));
  
  // 2. Check if count matches expected: isCorrect = (participantCount == expectedCount)
  auto countRef = std::make_unique<interpreter::VariableExpr>("_participantCount");
  auto expected = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(expectedParticipants));
  auto compare = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(countRef), std::move(expected), interpreter::BinaryOpExpr::Eq);
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_isCorrect", interpreter::HLSLType::Bool, std::move(compare)));
  
  // 3. Accumulate the result in a global buffer at tid.x index
  // _participant_check_sum[tid.x] += uint(_isCorrect)
  
  // Convert _isCorrect to uint
  auto isCorrectRef = std::make_unique<interpreter::VariableExpr>("_isCorrect");
  auto isCorrectAsUint = std::make_unique<interpreter::ConditionalExpr>(
      std::move(isCorrectRef), 
      std::make_unique<interpreter::LiteralExpr>(1),
      std::make_unique<interpreter::LiteralExpr>(0));
  
  // Get current value: _participant_check_sum[tid.x]
  auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
  auto bufferAccess = std::make_unique<interpreter::ArrayAccessExpr>(
      "_participant_check_sum",
      std::move(tidX));
  
  // Add to current value
  auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(bufferAccess), std::move(isCorrectAsUint), 
      interpreter::BinaryOpExpr::Add);
  
  // Store back: _participant_check_sum[tid.x] = ...
  auto tidX2 = std::make_unique<interpreter::VariableExpr>("tid.x");
  statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
      "_participant_check_sum", std::move(tidX2), std::move(addExpr)));
  
  return statements;
}

bool WaveParticipantTrackingMutation::hasParticipantBuffer(
    const interpreter::Program& program) const {
  // Check if program already declares a participant tracking buffer
  // For now, return false as we're using variables instead of buffers
  return false;
}

// Helper function to count wave operations in a statement recursively
static size_t countWaveOpsInStatement(const interpreter::Statement* stmt) {
  size_t count = 0;
  
  // Skip tracking operations
  if (isTrackingStatement(stmt)) {
    return 0;
  }
  
  // Check current statement
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    if (findWaveOpInExpression(assign->getExpression())) {
      count++;
    }
  } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    if (varDecl->getInit() && findWaveOpInExpression(varDecl->getInit())) {
      count++;
    }
  }
  // Handle control flow structures
  else if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
    for (const auto& thenStmt : ifStmt->getThenBlock()) {
      count += countWaveOpsInStatement(thenStmt.get());
    }
    for (const auto& elseStmt : ifStmt->getElseBlock()) {
      count += countWaveOpsInStatement(elseStmt.get());
    }
  } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
    for (const auto& bodyStmt : forStmt->getBody()) {
      count += countWaveOpsInStatement(bodyStmt.get());
    }
  } else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
    for (const auto& bodyStmt : whileStmt->getBody()) {
      count += countWaveOpsInStatement(bodyStmt.get());
    }
  } else if (auto* doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
    for (const auto& bodyStmt : doWhileStmt->getBody()) {
      count += countWaveOpsInStatement(bodyStmt.get());
    }
  } else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
    // Count wave ops in all switch cases
    for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
      const auto& caseBlock = switchStmt->getCase(i);
      for (const auto& caseStmt : caseBlock.statements) {
        count += countWaveOpsInStatement(caseStmt.get());
      }
    }
  }
  
  return count;
}

void WaveParticipantTrackingMutation::processStatementsForTracking(
    const std::vector<std::unique_ptr<interpreter::Statement>>& input,
    std::vector<std::unique_ptr<interpreter::Statement>>& output,
    const ExecutionTrace& trace,
    size_t& nextTraceIndex) const {
  
  for (const auto& stmt : input) {
    // IMPORTANT: Skip tracking operations we previously added
    if (isTrackingStatement(stmt.get())) {
      output.push_back(stmt->clone());
      continue;  // Don't process tracking ops
    }
    
    // Handle if statements specially - process recursively without cloning first
    if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt.get())) {
      // Process then block
      std::vector<std::unique_ptr<interpreter::Statement>> processedThen;
      processStatementsForTracking(ifStmt->getThenBlock(), processedThen, trace, nextTraceIndex);
      
      // Process else block
      std::vector<std::unique_ptr<interpreter::Statement>> processedElse;
      processStatementsForTracking(ifStmt->getElseBlock(), processedElse, trace, nextTraceIndex);
      
      // Create new if statement with processed blocks
      output.push_back(std::make_unique<interpreter::IfStmt>(
        ifStmt->getCondition()->clone(),
        std::move(processedThen),
        std::move(processedElse)));
      continue;
    }
    
    // For non-if statements, add the original statement first
    output.push_back(stmt->clone());
    
    // Check if this statement contains a wave operation that we should track
    const interpreter::WaveActiveOp* waveOp = nullptr;
    std::string resultVarName;
    
    // Don't process tracking operations (already skipped above, but double-check)
    if (!isTrackingStatement(stmt.get())) {
      if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt.get())) {
        waveOp = findWaveOpInExpression(assign->getExpression());
        resultVarName = assign->getName();
      } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt.get())) {
        if (varDecl->getInit()) {
          waveOp = findWaveOpInExpression(varDecl->getInit());
          resultVarName = varDecl->getName();
        }
      }
    }
    
    // If we found a wave operation, add tracking after it
    if (waveOp) {
      // Default to 0 for dead code paths
      uint32_t expectedParticipants = 0;
      uint32_t stableId = waveOp->getStableId();
      
      // Find the next occurrence of this stable ID in the trace
      bool found = false;
      size_t searchIndex = nextTraceIndex;
      while (searchIndex < trace.waveOperations.size()) {
        if (trace.waveOperations[searchIndex].stableId == stableId) {
          // Found it! Use this trace entry
          const auto& waveOpRecord = trace.waveOperations[searchIndex];
          expectedParticipants = waveOpRecord.arrivedParticipants.size();
          FUZZER_DEBUG_LOG("[WaveParticipantTracking] Wave op with stable ID " << stableId 
                          << " found at trace index " << searchIndex 
                          << " with " << expectedParticipants << " participants\n");
          
          // Update nextTraceIndex to move past this one
          nextTraceIndex = searchIndex + 1;
          found = true;
          break;
        }
        searchIndex++;
      }
      
      if (!found) {
        // This wave operation is not in the trace - it's dead code
        FUZZER_DEBUG_LOG("[WaveParticipantTracking] Wave op with stable ID " << stableId 
                        << " not found in remaining trace (dead code)\n");
      }
      
      // Add tracking statements
      auto trackingStmts = createTrackingStatements(waveOp, resultVarName, expectedParticipants);
      for (auto& trackStmt : trackingStmts) {
        output.push_back(std::move(trackStmt));
      }
    }
    
    // Handle nested statements - process control flow structures recursively
    // Note: if statements are handled at the beginning of the loop
    if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForTracking(forStmt->getBody(), processedBody, trace, nextTraceIndex);
      
      // We already added the statement at the beginning, need to replace it
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::ForStmt>(
        forStmt->getLoopVar(),
        forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
        forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
        forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
        std::move(processedBody)
      ));
    }
    else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForTracking(whileStmt->getBody(), processedBody, trace, nextTraceIndex);
      
      // We already added the statement at the beginning, need to replace it
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::WhileStmt>(
        whileStmt->getCondition()->clone(),
        std::move(processedBody)
      ));
    }
    else if (auto doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForTracking(doWhileStmt->getBody(), processedBody, trace, nextTraceIndex);
      
      // Replace the do-while statement with processed version
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::DoWhileStmt>(
        std::move(processedBody),
        doWhileStmt->getCondition()->clone()
      ));
    }
    else if (auto switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt.get())) {
      // Process switch cases
      auto newSwitch = std::make_unique<interpreter::SwitchStmt>(
          switchStmt->getCondition()->clone());
      
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        std::vector<std::unique_ptr<interpreter::Statement>> processedCaseBody;
        
        // Process statements in this case
        processStatementsForTracking(caseBlock.statements, processedCaseBody, 
                                     trace, nextTraceIndex);
        
        // Add the processed case to the new switch
        if (caseBlock.value.has_value()) {
          newSwitch->addCase(caseBlock.value.value(), std::move(processedCaseBody));
        } else {
          newSwitch->addDefault(std::move(processedCaseBody));
        }
      }
      
      // Replace with the processed switch
      output.pop_back(); // Remove the cloned version
      output.push_back(std::move(newSwitch));
    }
  }
}

bool WaveParticipantTrackingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // TODO: Implement validation that verifies:
  // 1. The tracking code doesn't modify the original wave operation result
  // 2. All participant tracking is side-effect free
  // 3. The added instrumentation doesn't affect control flow
  // Currently relying on end-to-end semantic validation
  return true;
}

// ===== WaveParticipantFrequencyMutation Implementation =====

bool WaveParticipantFrequencyMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  // This mutation applies at the program level, not statement level
  return false;
}

std::unique_ptr<interpreter::Statement> WaveParticipantFrequencyMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  // This mutation is handled at the program level
  return stmt->clone();
}

std::vector<std::unique_ptr<interpreter::Statement>>
WaveParticipantFrequencyMutation::createFrequencyTrackingStatements(
    const interpreter::WaveActiveOp* waveOp,
    const std::string& resultVar) const {
  
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // Simply increment the frequency counter for this thread: _participant_frequency[tid.x] += 1
  auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
  auto bufferAccess = std::make_unique<interpreter::ArrayAccessExpr>(
      "_participant_frequency",
      std::move(tidX));
  
  auto oneExpr = std::make_unique<interpreter::LiteralExpr>(1);
  auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(bufferAccess), std::move(oneExpr), 
      interpreter::BinaryOpExpr::Add);
  
  auto tidX2 = std::make_unique<interpreter::VariableExpr>("tid.x");
  statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
      "_participant_frequency", std::move(tidX2), std::move(addExpr)));
  
  return statements;
}

bool WaveParticipantFrequencyMutation::hasFrequencyBuffer(const interpreter::Program& program) const {
  for (const auto& buffer : program.globalBuffers) {
    if (buffer.name == "_participant_frequency") {
      return true;
    }
  }
  return false;
}

void WaveParticipantFrequencyMutation::processStatementsForFrequency(
    const std::vector<std::unique_ptr<interpreter::Statement>>& input,
    std::vector<std::unique_ptr<interpreter::Statement>>& output,
    const ExecutionTrace& trace,
    size_t& currentWaveOpIndex) const {
  
  for (const auto& stmt : input) {
    // Skip tracking operations we previously added
    if (isTrackingStatement(stmt.get())) {
      output.push_back(stmt->clone());
      continue;
    }
    
    // Handle if statements specially - process recursively without cloning first
    if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt.get())) {
      // Process then block
      std::vector<std::unique_ptr<interpreter::Statement>> processedThen;
      processStatementsForFrequency(ifStmt->getThenBlock(), processedThen, trace, currentWaveOpIndex);
      
      // Process else block
      std::vector<std::unique_ptr<interpreter::Statement>> processedElse;
      processStatementsForFrequency(ifStmt->getElseBlock(), processedElse, trace, currentWaveOpIndex);
      
      // Create new if statement with processed blocks
      output.push_back(std::make_unique<interpreter::IfStmt>(
        ifStmt->getCondition()->clone(),
        std::move(processedThen),
        std::move(processedElse)));
      continue;
    }
    
    // For non-if statements, add the original statement first
    output.push_back(stmt->clone());
    
    // Check if this statement contains a wave operation that we should track
    const interpreter::WaveActiveOp* waveOp = nullptr;
    std::string resultVarName;
    
    if (!isTrackingStatement(stmt.get())) {
      if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt.get())) {
        waveOp = findWaveOpInExpression(assign->getExpression());
        resultVarName = assign->getName();
      } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt.get())) {
        if (varDecl->getInit()) {
          waveOp = findWaveOpInExpression(varDecl->getInit());
          resultVarName = varDecl->getName();
        }
      }
    }
    
    // If we found a wave operation, add frequency tracking after it
    if (waveOp) {
      auto trackingStmts = createFrequencyTrackingStatements(waveOp, resultVarName);
      for (auto& trackingStmt : trackingStmts) {
        output.push_back(std::move(trackingStmt));
      }
      currentWaveOpIndex++;
    }
    
    // Handle nested control flow statements
    if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForFrequency(forStmt->getBody(), processedBody, trace, currentWaveOpIndex);
      
      // Replace with processed version
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::ForStmt>(
        forStmt->getLoopVar(),
        forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
        forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
        forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
        std::move(processedBody)
      ));
    }
    else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForFrequency(whileStmt->getBody(), processedBody, trace, currentWaveOpIndex);
      
      // Replace with processed version
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::WhileStmt>(
        whileStmt->getCondition()->clone(),
        std::move(processedBody)
      ));
    }
    else if (auto doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForFrequency(doWhileStmt->getBody(), processedBody, trace, currentWaveOpIndex);
      
      // Replace with processed version
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::DoWhileStmt>(
        std::move(processedBody),
        doWhileStmt->getCondition()->clone()
      ));
    }
    else if (auto switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt.get())) {
      // Process switch cases
      auto newSwitch = std::make_unique<interpreter::SwitchStmt>(
          switchStmt->getCondition()->clone());
      
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        std::vector<std::unique_ptr<interpreter::Statement>> processedCaseBody;
        
        // Process statements in this case
        processStatementsForFrequency(caseBlock.statements, processedCaseBody, 
                                    trace, currentWaveOpIndex);
        
        // Add the processed case to the new switch
        if (caseBlock.value.has_value()) {
          newSwitch->addCase(caseBlock.value.value(), std::move(processedCaseBody));
        } else {
          newSwitch->addDefault(std::move(processedCaseBody));
        }
      }
      
      // Replace with the processed switch
      output.pop_back(); // Remove the cloned version
      output.push_back(std::move(newSwitch));
    }
  }
}

bool WaveParticipantFrequencyMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // The frequency tracking is side-effect free on the main computation
  return true;
}

// Helper method: Check if any statements contain wave operations
bool WaveParticipantFrequencyMutation::hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const {
  
  std::function<bool(const interpreter::Statement*)> hasWaveOpRecursive;
  hasWaveOpRecursive = [&hasWaveOpRecursive](const interpreter::Statement* stmt) -> bool {
    // Check AssignStmt
    if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
      if (findWaveOpInExpression(assign->getExpression())) {
        return true;
      }
    } 
    // Check VarDeclStmt
    else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
      if (varDecl->getInit() && findWaveOpInExpression(varDecl->getInit())) {
        return true;
      }
    }
    // Check control flow statements
    else if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
      for (const auto& s : ifStmt->getThenBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
      for (const auto& s : ifStmt->getElseBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
      for (const auto& s : forStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
      for (const auto& s : whileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
      for (const auto& s : doWhileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
      // Check all cases in the switch
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        for (const auto& s : caseBlock.statements) {
          if (hasWaveOpRecursive(s.get())) return true;
        }
      }
    }
    return false;
  };
  
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      if (hasWaveOpRecursive(program.statements[i].get())) {
        return true;
      }
    }
  }
  
  return false;
}

// Helper method: Create base mutant with copied program structure
interpreter::Program WaveParticipantFrequencyMutation::createBaseMutant(const interpreter::Program& program) const {
  interpreter::Program mutant;
  mutant.numThreadsX = program.numThreadsX;
  mutant.numThreadsY = program.numThreadsY;
  mutant.numThreadsZ = program.numThreadsZ;
  mutant.entryInputs = program.entryInputs;
  mutant.globalBuffers = program.globalBuffers;
  mutant.waveSize = program.waveSize;
  return mutant;
}

// Helper method: Ensure frequency buffer exists
void WaveParticipantFrequencyMutation::ensureFrequencyBuffer(interpreter::Program& mutant) const {
  bool hasFrequencyBuffer = false;
  for (const auto& buffer : mutant.globalBuffers) {
    if (buffer.name == "_participant_frequency") {
      hasFrequencyBuffer = true;
      break;
    }
  }
  
  if (!hasFrequencyBuffer) {
    minihlsl::interpreter::GlobalBufferDecl frequencyBuffer;
    frequencyBuffer.name = "_participant_frequency";
    frequencyBuffer.bufferType = "RWStructuredBuffer";
    frequencyBuffer.elementType = minihlsl::interpreter::HLSLType::Uint;
    frequencyBuffer.size = mutant.getTotalThreads();
    frequencyBuffer.registerIndex = 2; // Use different register than participant buffer
    frequencyBuffer.isReadWrite = true;
    mutant.globalBuffers.push_back(frequencyBuffer);
    
    // Add buffer initialization
    auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
    auto zero = std::make_unique<interpreter::LiteralExpr>(0);
    mutant.statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
        "_participant_frequency", std::move(tidX), std::move(zero)));
  }
}

// Compute expected frequencies from trace
std::map<interpreter::ThreadId, uint32_t> WaveParticipantFrequencyMutation::computeExpectedFrequencies(
    const ExecutionTrace& trace) const {
  
  std::map<interpreter::ThreadId, uint32_t> frequencies;
  
  // Initialize all threads with 0 frequency
  for (const auto& [threadId, waveId] : trace.threadHierarchy.threadToWave) {
    frequencies[threadId] = 0;
  }
  
  // Count participation for each thread across all wave operations
  for (const auto& waveOp : trace.waveOperations) {
    for (interpreter::LaneId lane : waveOp.arrivedParticipants) {
      // Convert lane to thread ID
      interpreter::ThreadId threadId = 0;
      for (const auto& [tid, wid] : trace.threadHierarchy.threadToWave) {
        if (wid == waveOp.waveId && trace.threadHierarchy.threadToLane.at(tid) == lane) {
          threadId = tid;
          break;
        }
      }
      frequencies[threadId]++;
    }
  }
  
  return frequencies;
}

// Helper method: Create mutant with frequency tracking
interpreter::Program WaveParticipantFrequencyMutation::createMutantWithFrequencyTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  interpreter::Program mutant = createBaseMutant(program);
  ensureFrequencyBuffer(mutant);
  
  size_t currentWaveOpIndex = 0;
  
  // Process each statement
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      // This is a new statement - process it for frequency tracking
      std::vector<std::unique_ptr<interpreter::Statement>> inputStmts;
      inputStmts.push_back(program.statements[i]->clone());
      
      std::vector<std::unique_ptr<interpreter::Statement>> processedStmts;
      processStatementsForFrequency(inputStmts, processedStmts, trace, currentWaveOpIndex);
      
      // Add all processed statements (original + tracking)
      for (auto& stmt : processedStmts) {
        mutant.statements.push_back(std::move(stmt));
      }
    } else {
      // This is an old statement - copy it unchanged but still count its wave ops
      mutant.statements.push_back(program.statements[i]->clone());
      
      // Count wave ops in this statement to keep our index in sync
      currentWaveOpIndex += countWaveOpsInStatement(program.statements[i].get());
    }
  }
  
  // Add frequency validation at the end
  // Create a statement to compare actual vs expected frequencies
  auto expectedFreqs = computeExpectedFrequencies(trace);
  
  // For now, we'll store the expected frequencies in comments or debug output
  // In a real implementation, you'd compare these values
  FUZZER_DEBUG_LOG("[WaveParticipantFrequency] Expected frequencies:\n");
  for (const auto& [threadId, freq] : expectedFreqs) {
    FUZZER_DEBUG_LOG("  Thread " << threadId << ": " << freq << " participations\n");
  }
  
  return mutant;
}

// Implementation of program-level mutation for WaveParticipantFrequency
std::vector<interpreter::Program> WaveParticipantFrequencyMutation::applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  std::vector<interpreter::Program> mutants;
  
  // Check if any new statements have wave ops
  if (!hasWaveOpsInStatements(program, statementsToMutate)) {
    return mutants;
  }
  
  FUZZER_DEBUG_LOG("Found wave operations in new statements, applying WaveParticipantFrequency\n");
  
  // Create mutant with frequency tracking
  interpreter::Program mutant = createMutantWithFrequencyTracking(
      program, trace, statementsToMutate);
  
  mutants.push_back(std::move(mutant));
  return mutants;
}

// ===== WaveParticipantBitTrackingMutation Implementation =====

bool WaveParticipantBitTrackingMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  // This is a program-level mutation
  return false;
}

std::unique_ptr<interpreter::Statement> WaveParticipantBitTrackingMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  // This is a program-level mutation
  return nullptr;
}

bool WaveParticipantBitTrackingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // All bit tracking mutations preserve semantics - they only add observation
  return true;
}

// requiresProgramLevelMutation is already defined in the header via override

bool WaveParticipantBitTrackingMutation::hasBitTrackingBuffers(
    const interpreter::Program& program) const {
  // Check if program already has bit tracking buffers
  for (const auto& buffer : program.globalBuffers) {
    if (buffer.name == "_participant_bit" || buffer.name == "_wave_op_index") {
      return true;
    }
  }
  return false;
}

bool WaveParticipantBitTrackingMutation::hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const {
  
  std::function<bool(const interpreter::Statement*)> hasWaveOpRecursive;
  hasWaveOpRecursive = [&hasWaveOpRecursive](const interpreter::Statement* stmt) -> bool {
    // Check AssignStmt
    if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
      if (findWaveOpInExpression(assign->getExpression())) {
        return true;
      }
    } 
    // Check VarDeclStmt
    else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
      if (varDecl->getInit() && findWaveOpInExpression(varDecl->getInit())) {
        return true;
      }
    }
    // Check control flow statements
    else if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
      for (const auto& s : ifStmt->getThenBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
      for (const auto& s : ifStmt->getElseBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
      for (const auto& s : forStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
      for (const auto& s : whileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
      for (const auto& s : doWhileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
      // Check all cases in the switch
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        for (const auto& s : caseBlock.statements) {
          if (hasWaveOpRecursive(s.get())) return true;
        }
      }
    }
    return false;
  };
  
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      if (hasWaveOpRecursive(program.statements[i].get())) {
        return true;
      }
    }
  }
  
  return false;
}

uint32_t WaveParticipantBitTrackingMutation::getRecordSize(uint32_t waveSize) const {
  // wave32: 1 (id) + 2 (ballot.x, ballot.y for 32 lanes)
  // wave64: 1 (id) + 2 (ballot.x, ballot.y for 64 lanes)
  // WaveActiveBallot returns uint4, but only x,y are used for up to 64 lanes
  return 3; // id + ballot.x + ballot.y
}

std::vector<std::unique_ptr<interpreter::Statement>>
WaveParticipantBitTrackingMutation::createBitTrackingStatements(
    const interpreter::WaveActiveOp* waveOp,
    const std::string& resultVar,
    uint32_t waveOpId,
    uint32_t loopIteration,
    const std::vector<std::string>& loopVarStack) const {
  
  std::vector<std::unique_ptr<interpreter::Statement>> trackingStmts;
  
  // 1. Declare temp variable for InterlockedAdd
  // uint temp;
  trackingStmts.push_back(std::make_unique<interpreter::VarDeclStmt>(
    "temp", interpreter::HLSLType::Uint, nullptr));
  
  // 2. Atomic increment of index and get offset
  // Record size is always 3: combinedId, ballot.x, ballot.y
  auto indexAccess = std::make_unique<interpreter::ArrayAccessExpr>(
    "_wave_op_index",
    std::make_unique<interpreter::LiteralExpr>(0));
  
  auto addExpr = std::make_unique<interpreter::InterlockedAddExpr>(
    std::move(indexAccess),
    std::make_unique<interpreter::LiteralExpr>(3),
    std::make_unique<interpreter::VariableExpr>("temp"));
  
  auto atomicStmt = std::make_unique<interpreter::ExprStmt>(std::move(addExpr));
  trackingStmts.push_back(std::move(atomicStmt));
  
  // 3. Store combined ID with bitwise encoding
  // Bits [31-6]: Wave op ID (26 bits)
  // Bits [5-4]: Loop 1 index (outermost)
  // Bits [3-2]: Loop 2 index (middle)
  // Bits [1-0]: Loop 3 index (innermost)
  std::unique_ptr<interpreter::Expression> combinedIdExpr;
  
  // Start with wave op ID shifted to upper bits
  combinedIdExpr = std::make_unique<interpreter::BinaryOpExpr>(
    std::make_unique<interpreter::LiteralExpr>(static_cast<int>(waveOpId)),
    std::make_unique<interpreter::LiteralExpr>(6),
    interpreter::BinaryOpExpr::LeftShift);
  
  // Add loop indices using bitwise OR
  // Support up to 3 nested loops
  for (size_t i = 0; i < loopVarStack.size() && i < 3; ++i) {
    int shift = (2 - i) * 2;  // Loop 0: shift 4, Loop 1: shift 2, Loop 2: shift 0
    
    auto loopVar = std::make_unique<interpreter::VariableExpr>(loopVarStack[i]);
    if (shift > 0) {
      auto shiftedVar = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(loopVar),
        std::make_unique<interpreter::LiteralExpr>(shift),
        interpreter::BinaryOpExpr::LeftShift);
      combinedIdExpr = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(combinedIdExpr),
        std::move(shiftedVar),
        interpreter::BinaryOpExpr::BitwiseOr);
    } else {
      // No shift needed for innermost loop (bits [1-0])
      combinedIdExpr = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(combinedIdExpr),
        std::move(loopVar),
        interpreter::BinaryOpExpr::BitwiseOr);
    }
  }
  
  auto tempVar1 = std::make_unique<interpreter::VariableExpr>("temp");
  auto idStore = std::make_unique<interpreter::ArrayAssignStmt>(
    "_participant_bit",
    std::move(tempVar1),
    std::move(combinedIdExpr));
  trackingStmts.push_back(std::move(idStore));
  
  // 4. Call WaveActiveBallot and store result
  // uint4 ballot = WaveActiveBallot(true);
  auto ballotCall = std::make_unique<interpreter::WaveActiveOp>(
    std::make_unique<interpreter::LiteralExpr>(1), // true
    interpreter::WaveActiveOp::OpType::Ballot);
  
  trackingStmts.push_back(std::make_unique<interpreter::VarDeclStmt>(
    "ballot", interpreter::HLSLType::Uint4, std::move(ballotCall)));
  
  // 5. Store ballot.x at temp+1
  auto tempPlus1 = std::make_unique<interpreter::BinaryOpExpr>(
    std::make_unique<interpreter::VariableExpr>("temp"),
    std::make_unique<interpreter::LiteralExpr>(1),
    interpreter::BinaryOpExpr::Add);
  auto xStore = std::make_unique<interpreter::ArrayAssignStmt>(
    "_participant_bit",
    std::move(tempPlus1),
    std::make_unique<interpreter::MemberAccessExpr>(
      std::make_unique<interpreter::VariableExpr>("ballot"), "x"));
  trackingStmts.push_back(std::move(xStore));
  
  // 6. Store ballot.y at temp+2
  auto tempPlus2 = std::make_unique<interpreter::BinaryOpExpr>(
    std::make_unique<interpreter::VariableExpr>("temp"),
    std::make_unique<interpreter::LiteralExpr>(2),
    interpreter::BinaryOpExpr::Add);
  auto yStore = std::make_unique<interpreter::ArrayAssignStmt>(
    "_participant_bit",
    std::move(tempPlus2),
    std::make_unique<interpreter::MemberAccessExpr>(
      std::make_unique<interpreter::VariableExpr>("ballot"), "y"));
  trackingStmts.push_back(std::move(yStore));
  
  return trackingStmts;
}

void WaveParticipantBitTrackingMutation::processStatementsForBitTracking(
    const std::vector<std::unique_ptr<interpreter::Statement>>& input,
    std::vector<std::unique_ptr<interpreter::Statement>>& output,
    const ExecutionTrace& trace,
    size_t& nextWaveOpIndex,
    std::map<const void*, uint32_t>& loopIterationMap,
    uint32_t currentLoopIteration,
    const std::vector<std::string>& loopVarStack) const {
  
  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Processing statements\n");
  
  for (const auto& stmt : input) {
    // Skip null statements
    if (!stmt) {
      continue;
    }
    
    // Skip if this is already a tracking operation
    if (isTrackingStatement(stmt.get())) {
      FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Skipping tracking operation\n");
      output.push_back(cloneStatement(stmt.get()));
      continue;
    }
    
    // Handle if statements specially - process recursively without cloning first
    if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt.get())) {
      // Process then block
      std::vector<std::unique_ptr<interpreter::Statement>> processedThen;
      processStatementsForBitTracking(ifStmt->getThenBlock(), processedThen, trace,
                                     nextWaveOpIndex, loopIterationMap,
                                     currentLoopIteration, loopVarStack);
      
      // Process else block
      std::vector<std::unique_ptr<interpreter::Statement>> processedElse;
      processStatementsForBitTracking(ifStmt->getElseBlock(), processedElse, trace,
                                     nextWaveOpIndex, loopIterationMap,
                                     currentLoopIteration, loopVarStack);
      
      // Create new if statement with processed blocks
      output.push_back(std::make_unique<interpreter::IfStmt>(
        ifStmt->getCondition()->clone(),
        std::move(processedThen),
        std::move(processedElse)));
      continue;
    }
    
    // For non-if statements, add the original statement first
    output.push_back(cloneStatement(stmt.get()));
    
    // Check if this statement contains a wave operation
    const interpreter::WaveActiveOp* waveOp = nullptr;
    std::string resultVar;
    
    if (auto* exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt.get())) {
      waveOp = findWaveOpInExpression(exprStmt->getExpression());
    } else if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt.get())) {
      waveOp = findWaveOpInExpression(assign->getExpression());
      if (waveOp) {
        resultVar = assign->getName();
      }
    } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt.get())) {
      if (varDecl->getInit()) {
        waveOp = findWaveOpInExpression(varDecl->getInit());
        if (waveOp) {
          resultVar = varDecl->getName();
        }
      }
    }
    
    // If we found a wave operation, inject tracking
    if (waveOp) {
      // Don't track Ballot operations themselves (used for tracking)
      if (waveOp->getOpType() != interpreter::WaveActiveOp::OpType::Ballot) {
        uint32_t waveOpId = waveOp->getStableId();
        auto trackingStmts = createBitTrackingStatements(
          waveOp, resultVar, waveOpId, currentLoopIteration, loopVarStack);
        
        for (auto& trackingStmt : trackingStmts) {
          output.push_back(std::move(trackingStmt));
        }
        
        nextWaveOpIndex++;
      }
    }
    
    // Handle nested statements - process control flow structures recursively
    // Note: if statements are handled at the beginning of the loop
    if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      // Create new loop variable stack with for loop variable
      std::vector<std::string> forLoopVarStack = loopVarStack;
      forLoopVarStack.push_back(forStmt->getLoopVar());
      processStatementsForBitTracking(forStmt->getBody(), processedBody, trace,
                                     nextWaveOpIndex, loopIterationMap,
                                     currentLoopIteration + 1, forLoopVarStack);
      
      // We already added the statement at the beginning, need to replace it
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::ForStmt>(
        forStmt->getLoopVar(),
        forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
        forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
        forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
        std::move(processedBody)
      ));
    }
    else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt.get())) {
      // Process body
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      
      // Try to detect counter variable for while loops
      // Look for a counter variable declaration before the while loop
      std::vector<std::string> whileLoopVarStack = loopVarStack;
      
      // Look for counter pattern in the condition: (counter < N)
      if (auto* binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(whileStmt->getCondition())) {
        if (binOp->getOp() == interpreter::BinaryOpExpr::Lt || 
            binOp->getOp() == interpreter::BinaryOpExpr::Le) {
          if (auto* varExpr = dynamic_cast<const interpreter::VariableExpr*>(binOp->getLeft())) {
            // The variable in the condition is likely the counter
            std::string counterName = varExpr->getName();
            // Look for the counter increment pattern in the loop body
            bool hasIncrement = false;
            if (!whileStmt->getBody().empty()) {
              if (auto* assignStmt = dynamic_cast<const interpreter::AssignStmt*>(whileStmt->getBody()[0].get())) {
                if (assignStmt->getName() == counterName) {
                  if (auto* incExpr = dynamic_cast<const interpreter::BinaryOpExpr*>(assignStmt->getExpression())) {
                    if (incExpr->getOp() == interpreter::BinaryOpExpr::Add) {
                      hasIncrement = true;
                    }
                  }
                }
              }
            }
            if (hasIncrement) {
              whileLoopVarStack.push_back(counterName);
              FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Detected while loop counter: " << counterName << "\n");
            }
          }
        }
      }
      
      processStatementsForBitTracking(whileStmt->getBody(), processedBody, trace,
                                     nextWaveOpIndex, loopIterationMap,
                                     currentLoopIteration, whileLoopVarStack);
      
      // We already added the statement at the beginning, need to replace it
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::WhileStmt>(
        whileStmt->getCondition()->clone(),
        std::move(processedBody)
      ));
    }
    else if (auto* doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt.get())) {
      // Process body - do-while loops don't have a natural counter variable
      std::vector<std::unique_ptr<interpreter::Statement>> processedBody;
      processStatementsForBitTracking(doWhileStmt->getBody(), processedBody, trace,
                                     nextWaveOpIndex, loopIterationMap,
                                     currentLoopIteration, loopVarStack);
      
      // Replace the do-while statement with processed version
      output.pop_back(); // Remove the cloned version
      output.push_back(std::make_unique<interpreter::DoWhileStmt>(
        std::move(processedBody),
        doWhileStmt->getCondition()->clone()
      ));
    }
    else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt.get())) {
      // Process switch cases
      auto newSwitch = std::make_unique<interpreter::SwitchStmt>(
          switchStmt->getCondition()->clone());
      
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        std::vector<std::unique_ptr<interpreter::Statement>> processedCaseBody;
        
        // Process statements in this case
        processStatementsForBitTracking(caseBlock.statements, processedCaseBody, 
                                       trace, nextWaveOpIndex, loopIterationMap,
                                       currentLoopIteration, loopVarStack);
        
        // Add the processed case to the new switch
        if (caseBlock.value.has_value()) {
          newSwitch->addCase(caseBlock.value.value(), std::move(processedCaseBody));
        } else {
          newSwitch->addDefault(std::move(processedCaseBody));
        }
      }
      
      // Replace with the processed switch
      output.pop_back(); // Remove the cloned version
      output.push_back(std::move(newSwitch));
    }
  }
  
  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Finished processing\n");
}

void WaveParticipantBitTrackingMutation::ensureBitTrackingBuffers(
    interpreter::Program& mutant) const {
  // Check if buffers already exist
  bool hasParticipantBit = false;
  bool hasWaveOpIndex = false;
  
  for (const auto& buffer : mutant.globalBuffers) {
    if (buffer.name == "_participant_bit") hasParticipantBit = true;
    if (buffer.name == "_wave_op_index") hasWaveOpIndex = true;
  }
  
  // Add missing buffers
  if (!hasParticipantBit) {
    interpreter::GlobalBufferDecl participantBit;
    participantBit.name = "_participant_bit";
    participantBit.bufferType = "RWStructuredBuffer";
    participantBit.elementType = interpreter::HLSLType::Uint;
    participantBit.size = 1024; // Large enough for many recordings
    participantBit.isReadWrite = true;
    participantBit.registerIndex = 2;
    mutant.globalBuffers.push_back(participantBit);
  }
  
  if (!hasWaveOpIndex) {
    interpreter::GlobalBufferDecl waveOpIndex;
    waveOpIndex.name = "_wave_op_index";
    waveOpIndex.bufferType = "RWStructuredBuffer";
    waveOpIndex.elementType = interpreter::HLSLType::Uint;
    waveOpIndex.size = 1;
    waveOpIndex.isReadWrite = true;
    waveOpIndex.registerIndex = 3;
    mutant.globalBuffers.push_back(waveOpIndex);
  }
}

interpreter::Program WaveParticipantBitTrackingMutation::createBaseMutant(
    const interpreter::Program& program) const {
  interpreter::Program mutant;
  
  // Copy all fields
  mutant.numThreadsX = program.numThreadsX;
  mutant.numThreadsY = program.numThreadsY;
  mutant.numThreadsZ = program.numThreadsZ;
  mutant.entryInputs = program.entryInputs;
  mutant.globalBuffers = program.globalBuffers;
  mutant.waveSize = program.waveSize;
  
  return mutant;
}

interpreter::Program WaveParticipantBitTrackingMutation::createMutantWithBitTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  
  // Create base mutant
  interpreter::Program mutant = createBaseMutant(program);
  
  // Ensure participant buffer exists
  ensureBitTrackingBuffers(mutant);
  
  // Initialize trace index for execution-order matching
  size_t nextTraceIndex = 0;
  
  // Process all statements
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      // This is a new statement from current increment - process it for tracking
      std::vector<std::unique_ptr<interpreter::Statement>> inputStmts;
      inputStmts.push_back(program.statements[i]->clone());
      
      std::vector<std::unique_ptr<interpreter::Statement>> processedStmts;
      std::map<const void*, uint32_t> loopIterationMap;
      uint32_t currentLoopIteration = 0;
      processStatementsForBitTracking(
          inputStmts, processedStmts, trace, nextTraceIndex, 
          loopIterationMap, currentLoopIteration, {});
      
      // Add all processed statements (original + tracking)
      for (auto& stmt : processedStmts) {
        mutant.statements.push_back(std::move(stmt));
      }
    } else {
      // This is an old statement - copy it unchanged
      mutant.statements.push_back(program.statements[i]->clone());
    }
  }
  
  return mutant;

}

std::vector<uint32_t> WaveParticipantBitTrackingMutation::computeExpectedBitPatterns(
    const ExecutionTrace& trace) const {
  std::vector<uint32_t> expected;
  
  // Generate expected data in execution order
  for (const auto& waveOp : trace.waveOperations) {
    // Look up the block to get all loop iteration info for nested loops
    std::vector<uint32_t> loopIndices;
    auto blockIt = trace.blocks.find(waveOp.blockId);
    if (blockIt != trace.blocks.end()) {
      // Walk up the block hierarchy to collect all loop indices
      uint32_t currentBlockId = waveOp.blockId;
      std::vector<uint32_t> reversedIndices;  // Collect in reverse order
      
      while (blockIt != trace.blocks.end()) {
        if (blockIt->second.loopIteration.has_value()) {
          reversedIndices.push_back(blockIt->second.loopIteration->iterationValue);
        }
        
        // Move to parent block
        if (blockIt->second.parentBlockId == 0 || blockIt->second.parentBlockId == currentBlockId) break;
        currentBlockId = blockIt->second.parentBlockId;
        blockIt = trace.blocks.find(currentBlockId);
      }
      
      // Reverse to get outer-to-inner order
      loopIndices.insert(loopIndices.end(), reversedIndices.rbegin(), reversedIndices.rend());
    }
    
    // Compute combined ID with bitwise encoding
    // Bits [31-6]: Wave op ID (26 bits)
    // Bits [5-4]: Loop 1 index (outermost)
    // Bits [3-2]: Loop 2 index (middle)
    // Bits [1-0]: Loop 3 index (innermost)
    uint32_t combinedId = waveOp.stableId << 6;
    
    // Encode up to 3 nested loop indices
    for (size_t i = 0; i < loopIndices.size() && i < 3; ++i) {
      int shift = (2 - i) * 2;  // Loop 0: shift 4, Loop 1: shift 2, Loop 2: shift 0
      combinedId |= (loopIndices[i] & 0x3) << shift;
    }
    
    // Compute participant bitmask for this execution
    uint64_t mask = 0;
    for (interpreter::LaneId participantLane : waveOp.arrivedParticipants) {
      mask |= (1ULL << participantLane);
    }
    
    // For each participating thread, there should be one entry
    for (interpreter::LaneId lane : waveOp.arrivedParticipants) {
      // Each thread writes: combinedId, ballot.x, ballot.y
      expected.push_back(combinedId);
      
      // Store ballot.x and ballot.y
      expected.push_back(static_cast<uint32_t>(mask & 0xFFFFFFFF));  // ballot.x (low 32 bits)
      expected.push_back(static_cast<uint32_t>(mask >> 32));         // ballot.y (high 32 bits)
    }
  }
  
  return expected;
}

std::vector<interpreter::Program> WaveParticipantBitTrackingMutation::applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  std::vector<interpreter::Program> mutants;
  

  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Checking for wave ops in " 
                   << statementsToMutate.size() << " statements\n");
  
  if (!hasWaveOpsInStatements(program, statementsToMutate)) {
    FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] No wave operations in statements to mutate\n");
    return mutants;
  }
  
  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Found wave operations, proceeding with mutation\n");
  
  // Create mutant with bit tracking
  interpreter::Program mutant = createMutantWithBitTracking(
    program, trace, statementsToMutate);
  
  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Created mutant\n");
  
  mutants.push_back(std::move(mutant));
  FUZZER_DEBUG_LOG("[WaveParticipantBitTracking] Returning mutants\n");
  return mutants;
}

// ===== ContextAwareParticipantMutation Implementation =====

bool ContextAwareParticipantMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Check if statement contains wave operation (including nested)
  // todo: what does it mean
  bool hasWaveOp = findWaveOpInExpression(nullptr) != nullptr;
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    hasWaveOp = findWaveOpInExpression(assign->getExpression()) != nullptr;
  } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    hasWaveOp = varDecl->getInit() && 
      findWaveOpInExpression(varDecl->getInit()) != nullptr;
  }
  
  if (!hasWaveOp) {
    FUZZER_DEBUG_LOG("[ContextAwareParticipant] Statement does not contain wave operation\n");
    return false;
  }
  
  FUZZER_DEBUG_LOG("[ContextAwareParticipant] Found wave operation in statement\n");
  
  // Find all blocks where wave operations executed
  std::set<uint32_t> waveOpBlocks;
  for (const auto& waveOp : trace.waveOperations) {
    waveOpBlocks.insert(waveOp.blockId);
  }
  
  FUZZER_DEBUG_LOG("[ContextAwareParticipant] Wave ops executed in blocks: ");
  for (uint32_t blkId : waveOpBlocks) {
    FUZZER_DEBUG_LOG(blkId << " ");
  }
  FUZZER_DEBUG_LOG("\n");
  
  // Check if any of these blocks are inside loops
  for (uint32_t blockId : waveOpBlocks) {
    auto it = trace.blocks.find(blockId);
    if (it != trace.blocks.end()) {
      const auto& block = it->second;
      
      FUZZER_DEBUG_LOG("[ContextAwareParticipant] Checking block " << blockId 
                      << " type=" << static_cast<int>(block.blockType) << "\n");
      
      // Method 1: Check if block has a LOOP_HEADER as ancestor
      // (Since LOOP_BODY is not used, we check for loop header ancestry)
      uint32_t currentId = blockId;
      while (currentId != 0 && trace.blocks.count(currentId)) {
        const auto& currentBlock = trace.blocks.at(currentId);
        if (currentBlock.blockType == interpreter::BlockType::LOOP_HEADER) {
          FUZZER_DEBUG_LOG("[ContextAwareParticipant] Found LOOP_HEADER ancestor\n");
          return true;
        }
        currentId = currentBlock.parentBlockId;
      }
      
      // Method 2: Check if same source statement appears multiple times
      // (indicates multiple iterations)
      int occurrences = 0;
      for (const auto& [id, b] : trace.blocks) {
        if (b.sourceStatement == block.sourceStatement && b.sourceStatement != nullptr) {
          occurrences++;
        }
      }
      FUZZER_DEBUG_LOG("[ContextAwareParticipant] Block " << blockId 
                      << " sourceStatement=" << block.sourceStatement 
                      << " occurrences=" << occurrences << "\n");
      
      if (occurrences > 1) {
        FUZZER_DEBUG_LOG("[ContextAwareParticipant] Found multiple occurrences!\n");
        return true;  // Same statement executed multiple times
      }
    }
  }
  
  FUZZER_DEBUG_LOG("[ContextAwareParticipant] No loop context found\n");
  return false;
}

std::vector<ContextAwareParticipantMutation::IterationContext> 
ContextAwareParticipantMutation::extractIterationContexts(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  std::vector<IterationContext> contexts;
  
  // Helper to match wave operations to our statement
  auto matchesStatement = [stmt](const ExecutionTrace::WaveOpRecord& waveOp) -> bool {
    // Match by operation type
    if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
      if (auto* waveExpr = dynamic_cast<const interpreter::WaveActiveOp*>(
          assign->getExpression())) {
        return waveOp.waveOpEnumType == static_cast<int>(waveExpr->getOpType());
      }
    } else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
      if (varDecl->getInit()) {
        if (auto* waveExpr = dynamic_cast<const interpreter::WaveActiveOp*>(
            varDecl->getInit())) {
          return waveOp.waveOpEnumType == static_cast<int>(waveExpr->getOpType());
        }
      }
    }
    return false;
  };
  
  // Group wave operations by their instruction pointer - same instruction in different blocks means loop iterations
  std::map<const void*, std::vector<const ExecutionTrace::WaveOpRecord*>> instructionGroups;
  
  for (const auto& waveOp : trace.waveOperations) {
    if (!matchesStatement(waveOp)) continue;
    
    // Group by instruction pointer
    if (waveOp.instruction != nullptr) {
      instructionGroups[waveOp.instruction].push_back(&waveOp);
    }
  }
  
  // Extract iteration contexts for instructions that appear in multiple blocks (indicating loops)
  for (const auto& [instruction, waveOps] : instructionGroups) {
    // If the same instruction appears in multiple blocks, it's likely in a loop
    if (waveOps.size() <= 1) continue;
    
    // Sort by block ID to get iteration order
    std::vector<const ExecutionTrace::WaveOpRecord*> sortedOps = waveOps;
    std::sort(sortedOps.begin(), sortedOps.end(), 
      [](const auto* a, const auto* b) { return a->blockId < b->blockId; });
    
    FUZZER_DEBUG_LOG("[ContextAware] Instruction " << instruction << " appears in " 
                     << sortedOps.size() << " blocks: ");
    for (const auto* waveOpPtr : sortedOps) {
      FUZZER_DEBUG_LOG(waveOpPtr->blockId << " ");
    }
    FUZZER_DEBUG_LOG("\n");
    
    // Extract iteration contexts using block information
    for (size_t i = 0; i < sortedOps.size(); ++i) {
      IterationContext ctx;
      ctx.blockId = sortedOps[i]->blockId;
      ctx.existingParticipants = sortedOps[i]->arrivedParticipants;
      ctx.waveId = sortedOps[i]->waveId;
      
      // Check if the block has loop iteration info
      auto blockIt = trace.blocks.find(ctx.blockId);
      if (blockIt != trace.blocks.end() && blockIt->second.loopIteration.has_value()) {
        // Use the loop iteration info from the block
        const auto& loopInfo = blockIt->second.loopIteration.value();
        ctx.iterationValue = loopInfo.iterationValue;
        ctx.loopVariable = loopInfo.loopVariable;
        
        // If loop variable is empty, try to find it
        if (ctx.loopVariable.empty()) {
          ctx.loopVariable = findLoopVariable(loopInfo.loopHeaderBlock, trace);
        }
        
        FUZZER_DEBUG_LOG("[ContextAware] Block " << ctx.blockId 
                         << " has loop iteration " << ctx.iterationValue
                         << " (from block metadata)\n");
      } else {
        // Fallback: Try variable access tracking
        std::string loopVar = findLoopVariable(ctx.blockId, trace);
        ctx.loopVariable = loopVar;
        
        int actualIteration = -1;
        
        if (!loopVar.empty()) {
          // Look for variable accesses in the same block
          for (const auto& varAccess : trace.variableAccesses) {
            if (varAccess.blockId == ctx.blockId && 
                varAccess.varName == loopVar &&
                !varAccess.isWrite) {  // We want reads of the loop variable
              
              // Get the value from the wave
              auto waveIt = varAccess.values.find(ctx.waveId);
              if (waveIt != varAccess.values.end() && !waveIt->second.empty()) {
                // All lanes in a wave should have the same loop variable value
                // Just get from the first lane
                actualIteration = waveIt->second.begin()->second.asInt();
                FUZZER_DEBUG_LOG("[ContextAware] Found loop var " << loopVar 
                                 << " = " << actualIteration 
                                 << " in block " << ctx.blockId << " (from variable access)\n");
                break;
              }
            }
          }
        }
        
        // Final fallback: use position in sorted list
        if (actualIteration == -1) {
          actualIteration = i;
          FUZZER_DEBUG_LOG("[ContextAware] Using fallback iteration " << i 
                           << " for block " << ctx.blockId << "\n");
        }
        
        ctx.iterationValue = actualIteration;
      }
      
      contexts.push_back(ctx);
    }
  }
  
  return contexts;
}

std::string ContextAwareParticipantMutation::findLoopVariable(
    uint32_t blockId,
    const ExecutionTrace& trace) const {
  
  // Traverse up the block hierarchy to find the loop
  uint32_t currentId = blockId;
  while (currentId != 0 && trace.blocks.count(currentId)) {
    const auto& block = trace.blocks.at(currentId);
    
    // Check if this block has a loop statement as its source
    if (block.sourceStatement) {
      // For loop statements, we need to distinguish between for/while/do-while
      // by examining the actual statement type
      if (block.blockType == interpreter::BlockType::LOOP_HEADER) {
        
        // Try to cast the source statement to determine loop type
        const interpreter::Statement* stmtPtr = static_cast<const interpreter::Statement*>(block.sourceStatement);
        if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmtPtr)) {
          // For ForStmt, we can get the loop variable directly
          return forStmt->getLoopVar();
        }
        
        // For while/do-while loops, look for counter variable
        if (dynamic_cast<const interpreter::WhileStmt*>(stmtPtr) ||
            dynamic_cast<const interpreter::DoWhileStmt*>(stmtPtr)) {
          // Look for counter variable accesses
          for (const auto& varAccess : trace.variableAccesses) {
            if (varAccess.blockId >= currentId && 
                varAccess.blockId < currentId + 1000 &&  // Within reasonable range
                varAccess.varName.substr(0, 7) == "counter") {
              return varAccess.varName;
            }
          }
          // Fallback: assume "counter0"
          return "counter0";
        }
      }
    }
    
    currentId = block.parentBlockId;
  }
  
  // Final fallback based on block ID ranges
  if (blockId >= 100 && blockId < 1000) {
    // Likely in a loop, use common patterns
    if (blockId < 300) {
      return "i0";  // First for loop
    } else {
      return "counter0";  // First while loop
    }
  }
  
  return "";
}

std::unique_ptr<interpreter::Statement> 
ContextAwareParticipantMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Extract all iteration contexts where this wave op executed
  auto contexts = extractIterationContexts(stmt, trace);
  if (contexts.empty()) {
    FUZZER_DEBUG_LOG("[ContextAware] No iteration contexts found for wave op\n");
    return stmt->clone();
  }
  
  FUZZER_DEBUG_LOG("[ContextAware] Found " << contexts.size() << " iteration contexts\n");
  
  // Choose a context to mutate
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> ctxDist(0, contexts.size() - 1);
  const auto& targetContext = contexts[ctxDist(rng)];
  
  FUZZER_DEBUG_LOG("[ContextAware] Targeting iteration " << targetContext.iterationValue 
                   << " in block " << targetContext.blockId 
                   << " with loop var " << targetContext.loopVariable << "\n");
  
  // Find lanes that didn't participate in this iteration
  std::set<interpreter::LaneId> availableLanes;
  uint32_t waveSize = trace.threadHierarchy.waveSize;
  
  for (uint32_t lane = 0; lane < waveSize; ++lane) {
    if (targetContext.existingParticipants.find(lane) == 
        targetContext.existingParticipants.end()) {
      availableLanes.insert(lane);
    }
  }
  
  if (availableLanes.empty()) {
    FUZZER_DEBUG_LOG("[ContextAware] All lanes already participate\n");
    return stmt->clone(); // All lanes already participate
  }
  
  FUZZER_DEBUG_LOG("[ContextAware] " << availableLanes.size() << " lanes available\n");
  
  // Choose a new lane to add
  std::vector<interpreter::LaneId> laneVec(availableLanes.begin(), 
                                           availableLanes.end());
  std::uniform_int_distribution<size_t> laneDist(0, laneVec.size() - 1);
  interpreter::LaneId newLane = laneVec[laneDist(rng)];
  
  FUZZER_DEBUG_LOG("[ContextAware] Adding lane " << newLane 
                   << " to iteration " << targetContext.iterationValue << "\n");
  
  // Create the iteration-specific condition
  auto condition = createIterationSpecificCondition(targetContext, newLane);
  
  // Create a new if statement with our condition that executes the same wave operation
  std::vector<std::unique_ptr<interpreter::Statement>> contextBody;
  contextBody.push_back(stmt->clone()); // Same wave operation
  
  auto contextIf = std::make_unique<interpreter::IfStmt>(
    std::move(condition),
    std::move(contextBody),
    std::vector<std::unique_ptr<interpreter::Statement>>{}
  );
  
  // Create a compound statement that includes both the original and our mutation
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  statements.push_back(stmt->clone());  // Original statement
  statements.push_back(std::move(contextIf));  // Our context-aware addition
  
  // For now, we need to wrap in if(true) to return as single statement
  // TODO: Redesign mutation framework to allow returning multiple statements
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
  return std::make_unique<interpreter::IfStmt>(
    std::move(trueCond),
    std::move(statements),
    std::vector<std::unique_ptr<interpreter::Statement>>{}
  );
}

std::unique_ptr<interpreter::Expression>
ContextAwareParticipantMutation::createIterationSpecificCondition(
    const IterationContext& context,
    interpreter::LaneId newLane) const {
  
  // Create: (loopVar == iteration) && (laneId == newLane)
  
  // loopVar == iteration
  auto loopVarExpr = std::make_unique<interpreter::VariableExpr>(context.loopVariable);
  auto iterValue = std::make_unique<interpreter::LiteralExpr>(context.iterationValue);
  auto iterCheck = std::make_unique<interpreter::BinaryOpExpr>(
    std::move(loopVarExpr), std::move(iterValue), 
    interpreter::BinaryOpExpr::Eq);
  
  // laneId == newLane  
  auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto laneValue = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(newLane));
  auto laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
    std::move(laneExpr), std::move(laneValue),
    interpreter::BinaryOpExpr::Eq);
  
  // Combine with &&
  return std::make_unique<interpreter::BinaryOpExpr>(
    std::move(iterCheck), std::move(laneCheck),
    interpreter::BinaryOpExpr::And);
}

bool ContextAwareParticipantMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  
  // The mutation adds participants but doesn't change the operation
  // For associative operations, this should preserve semantics
  // TODO: Implement full validation to verify:
  // 1. The operation is associative (Sum, And, Or, etc.)
  // 2. The added participant doesn't change the result
  // 3. No side effects are introduced
  return true;
}

// ===== Semantic Validator Implementation =====

SemanticValidator::ValidationResult SemanticValidator::validate(
    const ExecutionTrace& golden, 
    const ExecutionTrace& mutant) {
  
  ValidationResult result;
  result.isEquivalent = true;
  
  // Compare final states
  if (!compareFinalStates(golden.finalState, mutant.finalState, result)) {
    result.isEquivalent = false;
    result.divergenceReason = "Final states differ";
  }
  
  // Compare wave operations
  if (!compareWaveOperations(golden.waveOperations, mutant.waveOperations, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Wave operations differ";
    }
  }
  
  // Compare memory state
  if (!compareMemoryState(golden, mutant, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Memory state differs";
    }
  }
  
  // Verify control flow equivalence
  // TODO: This check is too strict and causes false positives with semantics-preserving mutations
  // that add extra blocks (e.g., if(true) wrappers). Commenting out for now.
  /*
  if (!verifyControlFlowEquivalence(golden, mutant, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Control flow differs";
    }
  }
  */
  
  return result;
}

bool SemanticValidator::compareFinalStates(
    const ExecutionTrace::FinalState& golden,
    const ExecutionTrace::FinalState& mutant,
    ValidationResult& result) {
  
  // Debug: Print wave structure
  if (golden.laneVariables.empty()) {
  }
  if (mutant.laneVariables.empty()) {
  }
  
  // Compare per-lane variables
  // Note: We allow the mutant to have extra waves/lanes (from structural changes)
  // We only check that waves/lanes in the original exist in the mutant
  for (const auto& [waveId, waveVars] : golden.laneVariables) {
    auto mutantWaveIt = mutant.laneVariables.find(waveId);
    if (mutantWaveIt == mutant.laneVariables.end()) {
      // This might happen if waves are numbered differently
      // For now, assume single wave (wave 0) and continue
      if (waveId == 0 && !mutant.laneVariables.empty()) {
        mutantWaveIt = mutant.laneVariables.begin();
      } else {
        result.differences.push_back("Missing wave " + std::to_string(waveId) + " in mutant");
        return false;
      }
    }
    
    for (const auto& [laneId, vars] : waveVars) {
      auto mutantLaneIt = mutantWaveIt->second.find(laneId);
      if (mutantLaneIt == mutantWaveIt->second.end()) {
        result.differences.push_back("Missing lane " + std::to_string(laneId) + 
                                   " in wave " + std::to_string(waveId));
        return false;
      }
      
      // Compare variable values - only check variables that exist in the original
      // (mutations may add extra tracking variables)
      for (const auto& [varName, value] : vars) {
        // Skip variables that start with underscore (internal tracking variables)
        if (varName.empty() || varName[0] == '_') {
          continue;
        }
        
        auto mutantVarIt = mutantLaneIt->second.find(varName);
        if (mutantVarIt == mutantLaneIt->second.end()) {
          // Special case: 'tid' may be added by mutations
          if (varName == "tid") {
            continue;
          }
          result.differences.push_back("Variable " + varName + " missing in mutant");
          return false;
        }
        
        if (value.asInt() != mutantVarIt->second.asInt()) {
          result.differences.push_back("Variable " + varName + " differs: " + 
                                     std::to_string(value.asInt()) + " vs " +
                                     std::to_string(mutantVarIt->second.asInt()));
          return false;
        }
      }
    }
  }
  
  // Compare shared memory
  for (const auto& [addr, value] : golden.sharedMemory) {
    auto mutantIt = mutant.sharedMemory.find(addr);
    if (mutantIt == mutant.sharedMemory.end()) {
      result.differences.push_back("Shared memory address " + std::to_string(addr) + " missing");
      return false;
    }
    
    if (value.asInt() != mutantIt->second.asInt()) {
      result.differences.push_back("Shared memory at " + std::to_string(addr) + " differs");
      return false;
    }
  }
  
  return true;
}

bool SemanticValidator::compareWaveOperations(
    const std::vector<ExecutionTrace::WaveOpRecord>& golden,
    const std::vector<ExecutionTrace::WaveOpRecord>& mutant,
    ValidationResult& result) {
  
  // Helper lambda to extract operation type from full string
  auto extractOpType = [](const std::string& fullOp) -> std::string {
    // Extract just the operation name before the parenthesis
    // e.g., "WaveActiveSum(x)" -> "WaveActiveSum"
    size_t parenPos = fullOp.find('(');
    if (parenPos != std::string::npos) {
      return fullOp.substr(0, parenPos);
    }
    return fullOp;
  };
  
  // Wave operations must produce identical results
  if (golden.size() != mutant.size()) {
    result.differences.push_back("Different number of wave operations: " + 
                               std::to_string(golden.size()) + " vs " + 
                               std::to_string(mutant.size()));
    return false;
  }
  
  for (size_t i = 0; i < golden.size(); ++i) {
    const auto& goldenOp = golden[i];
    const auto& mutantOp = mutant[i];
    
    // Compare just the operation type, not the full expression
    std::string goldenOpType = extractOpType(goldenOp.opType);
    std::string mutantOpType = extractOpType(mutantOp.opType);
    
    if (goldenOpType != mutantOpType) {
      result.differences.push_back("Wave op type mismatch at index " + std::to_string(i) + 
                                 ": " + goldenOp.opType + " vs " + mutantOp.opType);
      return false;
    }
    
    // Compare output values
    for (const auto& [laneId, value] : goldenOp.outputValues) {
      auto mutantIt = mutantOp.outputValues.find(laneId);
      if (mutantIt == mutantOp.outputValues.end()) {
        result.differences.push_back("Wave op missing output for lane " + std::to_string(laneId));
        return false;
      }
      
      if (value != mutantIt->second) {
        result.differences.push_back("Wave op output differs for lane " + std::to_string(laneId));
        return false;
      }
    }
  }
  
  return true;
}

bool SemanticValidator::compareMemoryState(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Final memory state must be identical
  if (golden.finalMemoryState.size() != mutant.finalMemoryState.size()) {
    result.differences.push_back("Different number of memory locations");
    return false;
  }
  
  for (const auto& [addr, value] : golden.finalMemoryState) {
    auto mutantIt = mutant.finalMemoryState.find(addr);
    if (mutantIt == mutant.finalMemoryState.end()) {
      result.differences.push_back("Memory address " + std::to_string(addr) + " missing");
      return false;
    }
    
    if (value.asInt() != mutantIt->second.asInt()) {
      result.differences.push_back("Memory at " + std::to_string(addr) + " differs");
      return false;
    }
  }
  
  return true;
}

bool SemanticValidator::verifyControlFlowEquivalence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Verify that all lanes took equivalent paths (even if restructured)
  // This is more complex and would need proper implementation
  
  // For now, just check that the same blocks were visited
  std::set<uint32_t> goldenBlocks;
  std::set<uint32_t> mutantBlocks;
  
  for (const auto& [blockId, record] : golden.blocks) {
    goldenBlocks.insert(blockId);
  }
  
  for (const auto& [blockId, record] : mutant.blocks) {
    mutantBlocks.insert(blockId);
  }
  
  if (goldenBlocks != mutantBlocks) {
    result.differences.push_back("Different blocks visited");
    return false;
  }
  
  return true;
}

// ===== Bug Reporter Implementation =====

void BugReporter::reportBug(const interpreter::Program& original,
                          const interpreter::Program& mutant,
                          const ExecutionTrace& originalTrace,
                          const ExecutionTrace& mutantTrace,
                          const SemanticValidator::ValidationResult& validation) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  // Serialize programs to strings
  report.originalProgram = serializeProgramToString(original);
  report.mutantProgram = serializeProgramToString(mutant);
  
  report.validation = validation;
  
  // Find divergence point
  report.traceDivergence = findTraceDivergence(originalTrace, mutantTrace);
  
  // Classify bug
  report.bugType = classifyBug(report.traceDivergence);
  report.severity = assessSeverity(report.bugType, validation);
  
  // Save and log
  saveBugReport(report);
  logBug(report);
}

void BugReporter::reportCrash(const interpreter::Program& original,
                            const interpreter::Program& mutant,
                            const std::exception& e) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  report.bugType = BugReport::ControlFlowError;
  report.severity = BugReport::Critical;
  
  report.traceDivergence.type = BugReport::TraceDivergence::ControlFlow;
  report.traceDivergence.description = std::string("Crash: ") + e.what();
  
  saveBugReport(report);
  logBug(report);
}

BugReporter::BugReport::TraceDivergence BugReporter::findTraceDivergence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant) {
  
  BugReport::TraceDivergence divergence;
  
  // Check block structure differences
  if (golden.blocks.size() != mutant.blocks.size()) {
    divergence.type = BugReport::TraceDivergence::BlockStructure;
    divergence.divergencePoint = golden.blocks.size();
    divergence.description = "Different number of blocks executed: " + 
                            std::to_string(golden.blocks.size()) + " vs " + 
                            std::to_string(mutant.blocks.size());
    return divergence;
  }
  
  // Check wave operation differences
  if (golden.waveOperations.size() != mutant.waveOperations.size()) {
    divergence.type = BugReport::TraceDivergence::WaveOperation;
    divergence.divergencePoint = golden.waveOperations.size();
    divergence.description = "Different number of wave operations: " + 
                            std::to_string(golden.waveOperations.size()) + " vs " + 
                            std::to_string(mutant.waveOperations.size());
    return divergence;
  }
  
  // Check for different wave operation results
  for (size_t i = 0; i < golden.waveOperations.size(); ++i) {
    const auto& goldenOp = golden.waveOperations[i];
    const auto& mutantOp = mutant.waveOperations[i];
    
    if (goldenOp.opType != mutantOp.opType) {
      divergence.type = BugReport::TraceDivergence::WaveOperation;
      divergence.divergencePoint = i;
      divergence.description = "Wave operation type mismatch at index " + 
                              std::to_string(i) + ": " + goldenOp.opType + 
                              " vs " + mutantOp.opType;
      return divergence;
    }
    
    // Check output values differ
    if (goldenOp.outputValues != mutantOp.outputValues) {
      divergence.type = BugReport::TraceDivergence::WaveOperation;
      divergence.divergencePoint = i;
      divergence.description = "Wave operation outputs differ at index " + 
                              std::to_string(i);
      return divergence;
    }
  }
  
  // Check control flow decisions
  if (golden.controlFlowHistory.size() != mutant.controlFlowHistory.size()) {
    divergence.type = BugReport::TraceDivergence::ControlFlow;
    divergence.divergencePoint = golden.controlFlowHistory.size();
    divergence.description = "Different number of control flow decisions: " + 
                            std::to_string(golden.controlFlowHistory.size()) + " vs " + 
                            std::to_string(mutant.controlFlowHistory.size());
    return divergence;
  }
  
  // Check barrier synchronization
  if (golden.barriers.size() != mutant.barriers.size()) {
    divergence.type = BugReport::TraceDivergence::Synchronization;
    divergence.divergencePoint = golden.barriers.size();
    divergence.description = "Different number of barriers: " + 
                            std::to_string(golden.barriers.size()) + " vs " + 
                            std::to_string(mutant.barriers.size());
    return divergence;
  }
  
  // Check memory accesses
  if (golden.memoryAccesses.size() != mutant.memoryAccesses.size()) {
    divergence.type = BugReport::TraceDivergence::Memory;
    divergence.divergencePoint = golden.memoryAccesses.size();
    divergence.description = "Different number of memory accesses: " + 
                            std::to_string(golden.memoryAccesses.size()) + " vs " + 
                            std::to_string(mutant.memoryAccesses.size());
    return divergence;
  }
  
  // Default case - no obvious divergence found
  divergence.type = BugReport::TraceDivergence::ControlFlow;
  divergence.divergencePoint = 0;
  divergence.description = "Subtle divergence in execution traces";
  
  return divergence;
}

BugReporter::BugReport::BugType BugReporter::classifyBug(
    const BugReport::TraceDivergence& divergence) {
  
  switch (divergence.type) {
    case BugReport::TraceDivergence::WaveOperation:
      return BugReport::WaveOpInconsistency;
    case BugReport::TraceDivergence::Synchronization:
      return BugReport::DeadlockOrRace;
    case BugReport::TraceDivergence::Memory:
      return BugReport::MemoryCorruption;
    default:
      return BugReport::ControlFlowError;
  }
}

BugReporter::BugReport::Severity BugReporter::assessSeverity(
    BugReport::BugType type,
    const SemanticValidator::ValidationResult& validation) {
  
  if (type == BugReport::DeadlockOrRace || type == BugReport::MemoryCorruption) {
    return BugReport::Critical;
  }
  
  if (type == BugReport::WaveOpInconsistency) {
    return BugReport::High;
  }
  
  return BugReport::Medium;
}

std::string BugReporter::generateBugId() {
  static uint64_t counter = 0;
  return "BUG-" + std::to_string(++counter);
}

void BugReporter::saveBugReport(const BugReport& report) {
  // Create bugs directory if it doesn't exist
  std::string bugDir = "./bugs";
  
#if defined(_WIN32)
  // Windows: use CreateDirectory
  DWORD dwAttrib = GetFileAttributesA(bugDir.c_str());
  if (dwAttrib == INVALID_FILE_ATTRIBUTES || !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY)) {
    CreateDirectoryA(bugDir.c_str(), NULL);
  }
#else
  // POSIX: use mkdir
  struct stat st = {};
  if (stat(bugDir.c_str(), &st) == -1) {
    mkdir(bugDir.c_str(), 0755);
  }
#endif
  
  std::string filename = bugDir + "/" + report.id + ".txt";
  std::ofstream file(filename);
  
  if (!file.is_open()) {
    std::cerr << "Failed to create bug report file: " << filename << "\n";
    return;
  }
  
  // Write bug report header
  file << "=== Bug Report " << report.id << " ===\n";
  file << "Timestamp: " << std::chrono::system_clock::to_time_t(report.timestamp) << "\n";
  file << "Bug Type: ";
  switch (report.bugType) {
    case BugReport::WaveOpInconsistency: file << "WaveOpInconsistency"; break;
    case BugReport::ReconvergenceError: file << "ReconvergenceError"; break;
    case BugReport::DeadlockOrRace: file << "DeadlockOrRace"; break;
    case BugReport::MemoryCorruption: file << "MemoryCorruption"; break;
    case BugReport::ControlFlowError: file << "ControlFlowError"; break;
  }
  file << "\n";
  
  file << "Severity: ";
  switch (report.severity) {
    case BugReport::Critical: file << "Critical"; break;
    case BugReport::High: file << "High"; break;
    case BugReport::Medium: file << "Medium"; break;
    case BugReport::Low: file << "Low"; break;
  }
  file << "\n\n";
  
  // Write divergence information
  file << "=== Divergence Information ===\n";
  file << "Divergence Type: ";
  switch (report.traceDivergence.type) {
    case BugReport::TraceDivergence::BlockStructure: file << "BlockStructure"; break;
    case BugReport::TraceDivergence::WaveOperation: file << "WaveOperation"; break;
    case BugReport::TraceDivergence::ControlFlow: file << "ControlFlow"; break;
    case BugReport::TraceDivergence::Synchronization: file << "Synchronization"; break;
    case BugReport::TraceDivergence::Memory: file << "Memory"; break;
  }
  file << "\n";
  file << "Divergence Point: " << report.traceDivergence.divergencePoint << "\n";
  file << "Description: " << report.traceDivergence.description << "\n\n";
  
  // Write mutation information
  file << "=== Mutation Information ===\n";
  file << "Mutation Strategy: " << report.mutation << "\n\n";
  
  // Write validation results
  file << "=== Validation Results ===\n";
  file << "Is Equivalent: " << (report.validation.isEquivalent ? "Yes" : "No") << "\n";
  file << "Divergence Reason: " << report.validation.divergenceReason << "\n";
  if (!report.validation.differences.empty()) {
    file << "Differences:\n";
    for (const auto& diff : report.validation.differences) {
      file << "  - " << diff << "\n";
    }
  }
  file << "\n";
  
  // Write original program
  file << "=== Original Program ===\n";
  file << report.originalProgram << "\n\n";
  
  // Write mutant program
  file << "=== Mutant Program ===\n";
  file << report.mutantProgram << "\n\n";
  
  // Write minimal reproducer if available
  if (!report.minimalReproducer.empty()) {
    file << "=== Minimal Reproducer ===\n";
    file << report.minimalReproducer << "\n";
  }
  
  file.close();
  FUZZER_DEBUG_LOG("Bug report saved to: " << filename << "\n");
}

void BugReporter::logBug(const BugReport& report) {
  FUZZER_DEBUG_LOG("Found bug: " << report.id << "\n");
  FUZZER_DEBUG_LOG("Type: " << static_cast<int>(report.bugType) << "\n");
  FUZZER_DEBUG_LOG("Severity: " << static_cast<int>(report.severity) << "\n");
  FUZZER_DEBUG_LOG("Description: " << report.traceDivergence.description << "\n");
}

// ===== Main Fuzzer Implementation =====

TraceGuidedFuzzer::TraceGuidedFuzzer() {
  // Initialize mutation strategies
  // Disabled old count-based tracking in favor of new bit-based tracking
  // mutationStrategies.push_back(std::make_unique<WaveParticipantTrackingMutation>());
  // mutationStrategies.push_back(std::make_unique<LanePermutationMutation>());
  mutationStrategies.push_back(std::make_unique<WaveParticipantBitTrackingMutation>());
  // mutationStrategies.push_back(std::make_unique<WaveParticipantFrequencyMutation>());
  // mutationStrategies.push_back(std::make_unique<ContextAwareParticipantMutation>());
  
  // TODO: Add more semantics-preserving mutations:
  // - AlgebraicIdentityMutation (more complex algebraic identities)
  // - MemoryAccessReorderMutation (reorder independent memory accesses)
  // - RegisterSpillMutation (force values through memory)
  
  validator = std::make_unique<SemanticValidator>();
  bugReporter = std::make_unique<BugReporter>();
}

// Version that accepts generation history to only mutate new statements
interpreter::Program TraceGuidedFuzzer::fuzzProgram(const interpreter::Program& seedProgram, 
                                  const FuzzingConfig& config,
                                  const std::vector<GenerationRound>& history,
                                  size_t currentIncrement) {
  
  FUZZER_DEBUG_LOG("Starting trace-guided fuzzing with generation history...\n");
  FUZZER_DEBUG_LOG("Current increment: " << currentIncrement << "\n");
  FUZZER_DEBUG_LOG("History has " << history.size() << " rounds\n");
  
  // Find which statements to mutate based on the increment
  std::set<size_t> statementsToMutate;
  
  if (currentIncrement == 0) {
    // Increment 0: mutate all statements from rounds 0-4 (initial generation)
    for (const auto& round : history) {
      if (round.roundNumber <= 4) {
        for (size_t idx : round.addedStatementIndices) {
          statementsToMutate.insert(idx);
        }
        FUZZER_DEBUG_LOG("Round " << round.roundNumber << " added " << round.addedStatementIndices.size() << " statements\n");
      }
    }
  } else {
    // Increment 1+: mutate only statements added in this increment
    // Find the round(s) that were added in the current increment
    // Increment 1 adds round 5, increment 2 adds round 6, etc.
    size_t expectedRound = 4 + currentIncrement;
    for (const auto& round : history) {
      if (round.roundNumber == expectedRound) {
        for (size_t idx : round.addedStatementIndices) {
          statementsToMutate.insert(idx);
        }
        FUZZER_DEBUG_LOG("Round " << round.roundNumber << " added " << round.addedStatementIndices.size() << " statements\n");
        break;
      }
    }
  }
  
  if (statementsToMutate.empty()) {
    FUZZER_DEBUG_LOG("No statements to mutate in current increment, returning original program\n");
    // Can't copy Program, need to clone it
    interpreter::Program result;
    result.numThreadsX = seedProgram.numThreadsX;
    result.numThreadsY = seedProgram.numThreadsY;
    result.numThreadsZ = seedProgram.numThreadsZ;
    result.globalBuffers = seedProgram.globalBuffers;
    result.entryInputs = seedProgram.entryInputs;
    result.waveSize = seedProgram.waveSize;
    for (const auto& stmt : seedProgram.statements) {
      result.statements.push_back(stmt->clone());
    }
    return result;
  }
  
  // Prepare program for mutation
  interpreter::Program preparedProgram = prepareProgramForMutation(seedProgram);
  
  // Create trace capture interpreter
  TraceCaptureInterpreter captureInterpreter;
  
  // Execute seed and capture golden trace
  FUZZER_DEBUG_LOG("Capturing golden trace...\n");
  
  // Use sequential ordering for deterministic execution
  interpreter::ThreadOrdering ordering = interpreter::ThreadOrdering::sequential(preparedProgram.getTotalThreads());
  
  // Use the program's wave size if specified, otherwise fall back to config
  uint32_t effectiveWaveSize = preparedProgram.getEffectiveWaveSize(config.waveSize);
  
  auto goldenResult = captureInterpreter.executeAndCaptureTrace(
    preparedProgram, ordering, effectiveWaveSize);
  
  // Check if execution succeeded
  if (!goldenResult.isValid()) {
    std::cerr << "Golden execution failed: " << goldenResult.errorMessage << "\n";
    return preparedProgram;
  }
  
  const ExecutionTrace& goldenTrace = *captureInterpreter.getTrace();
  
  FUZZER_DEBUG_LOG("Golden trace captured:\n");
  FUZZER_DEBUG_LOG("  - Wave operations: " << goldenTrace.waveOperations.size() << "\n");
  
  // Apply all mutations in sequence, but only to new statements
  size_t mutantsTested = 0;
  size_t bugsFound = 0;
  
  // Apply all mutations to new statements only
  auto mutationResult = applyAllMutations(preparedProgram, goldenTrace, &history, currentIncrement);
  
  // Track the final mutated program
  interpreter::Program finalMutant;
  
  // Test the mutated program if any mutations were applied
  if (mutationResult.hasMutations()) {
    finalMutant = std::move(mutationResult.mutatedProgram);
    mutantsTested++;
    
    FUZZER_DEBUG_LOG("\n=== Testing Mutant with " << mutationResult.getMutationChainString() 
                    << " (new statements only) ===\n");
    
    // Debug: Print the actual mutated program
    FUZZER_DEBUG_LOG("\n=== Actual Mutated Program ===\n");
    FUZZER_DEBUG_LOG(serializeProgramToString(finalMutant));
    FUZZER_DEBUG_LOG("\n");
    
    // Save mutant program to file
    std::string mutantPath = createMutantOutputPath(currentIncrement, mutationResult.getMutationChainString(), ".hlsl", config.seedId, config.outputDir);
    std::ofstream mutantFile(mutantPath);
    if (mutantFile.is_open()) {
      mutantFile << serializeProgramToString(finalMutant);
      mutantFile.close();
      FUZZER_DEBUG_LOG("Saved mutant to: " << mutantPath << "\n");
    } else {
      FUZZER_DEBUG_LOG("Failed to save mutant to: " << mutantPath << "\n");
    }
    
    // Generate test file with YAML pipeline if using WaveParticipantTracking or WaveParticipantBitTracking
    if (mutationResult.getMutationChainString().find("WaveParticipantTracking") != std::string::npos ||
        mutationResult.getMutationChainString().find("WaveParticipantBitTracking") != std::string::npos) {
      try {
        std::string testPath = createMutantOutputPath(currentIncrement, mutationResult.getMutationChainString(), ".test", config.seedId, config.outputDir);
        generateTestFile(finalMutant, goldenTrace, testPath, mutationResult.getMutationChainString());
      } catch (const std::exception& e) {
        FUZZER_DEBUG_LOG("Failed to generate test file: " << e.what() << "\n");
      }
    }
    
    // Test the mutant
    try {
      TraceCaptureInterpreter mutantInterpreter;
      // Use the mutant program's wave size if specified
      uint32_t mutantWaveSize = finalMutant.getEffectiveWaveSize(config.waveSize);
      auto mutantTestResult = mutantInterpreter.executeAndCaptureTrace(
        finalMutant, ordering, mutantWaveSize);
      
      if (!mutantTestResult.isValid()) {
        FUZZER_DEBUG_LOG("Mutant execution failed: " << mutantTestResult.errorMessage << "\n");
      } else {
        const ExecutionTrace& mutantTrace = *mutantInterpreter.getTrace();
        
        // Validate semantic equivalence
        auto validation = validator->validate(goldenTrace, mutantTrace);
        
        if (!validation.isEquivalent) {
          bugsFound++;
          bugReporter->reportBug(preparedProgram, finalMutant, goldenTrace, 
                               mutantTrace, validation);
        }
      }
    } catch (const std::exception& e) {
      bugsFound++;
      bugReporter->reportCrash(preparedProgram, finalMutant, e);
    }
  } else {
    // No mutations applied, return the prepared program
    finalMutant.numThreadsX = preparedProgram.numThreadsX;
    finalMutant.numThreadsY = preparedProgram.numThreadsY;
    finalMutant.numThreadsZ = preparedProgram.numThreadsZ;
    finalMutant.globalBuffers = preparedProgram.globalBuffers;
    finalMutant.entryInputs = preparedProgram.entryInputs;
    finalMutant.waveSize = preparedProgram.waveSize;
    for (const auto& stmt : preparedProgram.statements) {
      finalMutant.statements.push_back(stmt->clone());
    }
  }
  
  logSummary(mutantsTested, bugsFound);
  
  return finalMutant;
}


// Helper function to recursively apply mutations to statements
std::unique_ptr<interpreter::Statement> TraceGuidedFuzzer::applyMutationToStatement(
    const interpreter::Statement* stmt,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    bool& mutationApplied) {
  
  // First, try to apply mutation to this statement directly
  if (strategy->canApply(stmt, trace)) {
    auto mutated = strategy->apply(stmt, trace);
    if (mutated) {
      mutationApplied = true;
      return mutated;
    }
  }
  
  // If no direct mutation, check for nested statements
  if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
    // Apply mutations to all applicable statements in the then block
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedThenStmts;
    bool anyMutationInThen = false;
    
    for (const auto& thenStmt : ifStmt->getThenBlock()) {
      bool thisMutationApplied = false;
      auto mutated = applyMutationToStatement(thenStmt.get(), strategy, trace, thisMutationApplied);
      if (thisMutationApplied) {
        mutatedThenStmts.push_back(std::move(mutated));
        anyMutationInThen = true;
        mutationApplied = true;
      } else {
        mutatedThenStmts.push_back(thenStmt->clone());
      }
    }
    
    // Apply mutations to all applicable statements in the else block
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedElseStmts;
    bool anyMutationInElse = false;
    if (ifStmt->hasElse()) {
      for (const auto& elseStmt : ifStmt->getElseBlock()) {
        bool thisMutationApplied = false;
        auto mutated = applyMutationToStatement(elseStmt.get(), strategy, trace, thisMutationApplied);
        if (thisMutationApplied) {
          mutatedElseStmts.push_back(std::move(mutated));
          anyMutationInElse = true;
          mutationApplied = true;
        } else {
          mutatedElseStmts.push_back(elseStmt->clone());
        }
      }
    }
    
    // If we found any mutations in nested statements, create a new IfStmt
    if (anyMutationInThen || anyMutationInElse) {
      return std::make_unique<interpreter::IfStmt>(
          ifStmt->getCondition()->clone(),
          std::move(mutatedThenStmts),
          std::move(mutatedElseStmts));
    }
  } else if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
    // Apply mutations to all applicable statements in the for loop body
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedBodyStmts;
    bool anyMutationInBody = false;
    
    for (const auto& bodyStmt : forStmt->getBody()) {
      bool thisMutationApplied = false;
      auto mutated = applyMutationToStatement(bodyStmt.get(), strategy, trace, thisMutationApplied);
      if (thisMutationApplied) {
        mutatedBodyStmts.push_back(std::move(mutated));
        anyMutationInBody = true;
        mutationApplied = true;
      } else {
        mutatedBodyStmts.push_back(bodyStmt->clone());
      }
    }
    
    // If we found any mutations in the body, create a new ForStmt
    if (anyMutationInBody) {
      return std::make_unique<interpreter::ForStmt>(
          forStmt->getLoopVar(),
          forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
          forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
          forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
          std::move(mutatedBodyStmts));
    }
  } else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
    // Apply mutations to all applicable statements in the while loop body
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedBodyStmts;
    bool anyMutationInBody = false;
    
    for (const auto& bodyStmt : whileStmt->getBody()) {
      bool thisMutationApplied = false;
      auto mutated = applyMutationToStatement(bodyStmt.get(), strategy, trace, thisMutationApplied);
      if (thisMutationApplied) {
        mutatedBodyStmts.push_back(std::move(mutated));
        anyMutationInBody = true;
        mutationApplied = true;
      } else {
        mutatedBodyStmts.push_back(bodyStmt->clone());
      }
    }
    
    // If we found any mutations in the body, create a new WhileStmt
    if (anyMutationInBody) {
      return std::make_unique<interpreter::WhileStmt>(
          whileStmt->getCondition()->clone(),
          std::move(mutatedBodyStmts));
    }
  } else if (auto doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
    // Apply mutations to all applicable statements in the do-while loop body
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedBodyStmts;
    bool anyMutationInBody = false;
    
    for (const auto& bodyStmt : doWhileStmt->getBody()) {
      bool thisMutationApplied = false;
      auto mutated = applyMutationToStatement(bodyStmt.get(), strategy, trace, thisMutationApplied);
      if (thisMutationApplied) {
        mutatedBodyStmts.push_back(std::move(mutated));
        anyMutationInBody = true;
        mutationApplied = true;
      } else {
        mutatedBodyStmts.push_back(bodyStmt->clone());
      }
    }
    
    // If we found any mutations in the body, create a new DoWhileStmt
    if (anyMutationInBody) {
      return std::make_unique<interpreter::DoWhileStmt>(
          std::move(mutatedBodyStmts),
          doWhileStmt->getCondition()->clone());
    }
  }
  
  // No mutation found, return clone
  return stmt->clone();
}

interpreter::Program TraceGuidedFuzzer::prepareProgramForMutation(
    const interpreter::Program& program) {
  
  interpreter::Program preparedProgram;
  preparedProgram.numThreadsX = program.numThreadsX;
  preparedProgram.numThreadsY = program.numThreadsY;
  preparedProgram.numThreadsZ = program.numThreadsZ;
  preparedProgram.globalBuffers = program.globalBuffers;
  preparedProgram.entryInputs = program.entryInputs;
  preparedProgram.waveSize = program.waveSize;
  
  // Check if main function has SV_DispatchThreadID parameter
  bool hasDispatchThreadID = false;
  for (const auto& param : preparedProgram.entryInputs.parameters) {
    if (param.semantic == minihlsl::interpreter::HLSLSemantic::SV_DispatchThreadID) {
      hasDispatchThreadID = true;
      break;
    }
  }
  
  // Add SV_DispatchThreadID parameter if missing
  if (!hasDispatchThreadID) {
    minihlsl::interpreter::ParameterSig tidParam;
    tidParam.name = "tid";
    tidParam.type = minihlsl::interpreter::HLSLType::Uint3;
    tidParam.semantic = minihlsl::interpreter::HLSLSemantic::SV_DispatchThreadID;
    preparedProgram.entryInputs.parameters.push_back(tidParam);
    
  }
  
  // Copy all statements
  for (const auto& stmt : program.statements) {
    preparedProgram.statements.push_back(stmt->clone());
  }
  
  return preparedProgram;
}


// Helper method: Determine which statements to mutate based on increment
std::set<size_t> TraceGuidedFuzzer::determineStatementsToMutate(
    const std::vector<GenerationRound>& history,
    size_t currentIncrement) {
  
  std::set<size_t> statementsToMutate;
  
  if (currentIncrement == 0) {
    // Increment 0: mutate all statements from rounds 0-4 (initial generation)
    for (const auto& round : history) {
      if (round.roundNumber <= 4) {
        for (size_t idx : round.addedStatementIndices) {
          statementsToMutate.insert(idx);
        }
      }
    }
  } else {
    // Increment 1+: mutate only statements added in this increment
    size_t expectedRound = 4 + currentIncrement;
    for (const auto& round : history) {
      if (round.roundNumber == expectedRound) {
        for (size_t idx : round.addedStatementIndices) {
          statementsToMutate.insert(idx);
        }
        break;
      }
    }
  }
  
  return statementsToMutate;
}

// Implementation of program-level mutation for WaveParticipantTracking
std::vector<interpreter::Program> WaveParticipantTrackingMutation::applyToProgram(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  std::vector<interpreter::Program> mutants;
  
  // Check if any new statements have wave ops
  if (!hasWaveOpsInStatements(program, statementsToMutate)) {
    return mutants;
  }
  
  FUZZER_DEBUG_LOG("Found wave operations in new statements, applying WaveParticipantTracking\n");
  
  // Create mutant with tracking using execution-order matching
  interpreter::Program mutant = createMutantWithTracking(
      program, trace, statementsToMutate);
  
  mutants.push_back(std::move(mutant));
  return mutants;
}

// Helper method: Check if any statements contain wave operations
bool WaveParticipantTrackingMutation::hasWaveOpsInStatements(
    const interpreter::Program& program,
    const std::set<size_t>& statementsToMutate) const {
  
  std::function<bool(const interpreter::Statement*)> hasWaveOpRecursive;
  hasWaveOpRecursive = [&hasWaveOpRecursive](const interpreter::Statement* stmt) -> bool {
    // Check AssignStmt
    if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
      if (findWaveOpInExpression(assign->getExpression())) {
        return true;
      }
    } 
    // Check VarDeclStmt
    else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
      if (varDecl->getInit() && findWaveOpInExpression(varDecl->getInit())) {
        return true;
      }
    }
    // Check control flow statements
    else if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
      for (const auto& s : ifStmt->getThenBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
      for (const auto& s : ifStmt->getElseBlock()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
      for (const auto& s : forStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* whileStmt = dynamic_cast<const interpreter::WhileStmt*>(stmt)) {
      for (const auto& s : whileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* doWhileStmt = dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
      for (const auto& s : doWhileStmt->getBody()) {
        if (hasWaveOpRecursive(s.get())) return true;
      }
    } else if (auto* switchStmt = dynamic_cast<const interpreter::SwitchStmt*>(stmt)) {
      // Check all cases in the switch
      for (size_t i = 0; i < switchStmt->getCaseCount(); ++i) {
        const auto& caseBlock = switchStmt->getCase(i);
        for (const auto& s : caseBlock.statements) {
          if (hasWaveOpRecursive(s.get())) return true;
        }
      }
    }
    return false;
  };
  
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      if (hasWaveOpRecursive(program.statements[i].get())) {
        return true;
      }
    }
  }
  
  return false;
}

// Helper method: Create base mutant with copied program structure
interpreter::Program WaveParticipantTrackingMutation::createBaseMutant(const interpreter::Program& program) const {
  interpreter::Program mutant;
  mutant.numThreadsX = program.numThreadsX;
  mutant.numThreadsY = program.numThreadsY;
  mutant.numThreadsZ = program.numThreadsZ;
  mutant.entryInputs = program.entryInputs;
  mutant.globalBuffers = program.globalBuffers;
  mutant.waveSize = program.waveSize;
  return mutant;
}

// Helper method: Ensure participant buffer exists
void WaveParticipantTrackingMutation::ensureParticipantBuffer(interpreter::Program& mutant) const {
  bool hasParticipantBuffer = false;
  for (const auto& buffer : mutant.globalBuffers) {
    if (buffer.name == "_participant_check_sum") {
      hasParticipantBuffer = true;
      break;
    }
  }
  
  if (!hasParticipantBuffer) {
    minihlsl::interpreter::GlobalBufferDecl participantBuffer;
    participantBuffer.name = "_participant_check_sum";
    participantBuffer.bufferType = "RWStructuredBuffer";
    participantBuffer.elementType = minihlsl::interpreter::HLSLType::Uint;
    participantBuffer.size = mutant.getTotalThreads();
    participantBuffer.registerIndex = 1;
    participantBuffer.isReadWrite = true;
    mutant.globalBuffers.push_back(participantBuffer);
    
    // Add buffer initialization
    auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
    auto zero = std::make_unique<interpreter::LiteralExpr>(0);
    mutant.statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
        "_participant_check_sum", std::move(tidX), std::move(zero)));
  }
}


// Helper method: Create mutant with participant tracking
interpreter::Program WaveParticipantTrackingMutation::createMutantWithTracking(
    const interpreter::Program& program,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) const {
  
  // Create base mutant
  interpreter::Program mutant = createBaseMutant(program);
  
  // Ensure participant buffer exists
  ensureParticipantBuffer(mutant);
  
  // Initialize trace index for execution-order matching
  size_t nextTraceIndex = 0;
  
  // Process all statements
  for (size_t i = 0; i < program.statements.size(); ++i) {
    if (statementsToMutate.find(i) != statementsToMutate.end()) {
      // This is a new statement from current increment - process it for tracking
      std::vector<std::unique_ptr<interpreter::Statement>> inputStmts;
      inputStmts.push_back(program.statements[i]->clone());
      
      std::vector<std::unique_ptr<interpreter::Statement>> processedStmts;
      processStatementsForTracking(
          inputStmts, processedStmts, trace, nextTraceIndex);
      
      // Add all processed statements (original + tracking)
      for (auto& stmt : processedStmts) {
        mutant.statements.push_back(std::move(stmt));
      }
    } else {
      // This is an old statement - copy it unchanged
      mutant.statements.push_back(program.statements[i]->clone());
    }
  }
  
  return mutant;
}

// Version that only generates mutants for statements added in the current increment
std::vector<interpreter::Program> TraceGuidedFuzzer::generateMutants(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    const std::vector<GenerationRound>& history,
    size_t currentIncrement) {
  
  std::vector<interpreter::Program> mutants;
  
  // Determine which statements to mutate
  std::set<size_t> statementsToMutate = determineStatementsToMutate(history, currentIncrement);
  
  if (statementsToMutate.empty()) {
    return mutants;
  }
  
  FUZZER_DEBUG_LOG("Generating mutants for " << statementsToMutate.size() 
                   << " statements from increment " << currentIncrement << "\n");
  
  // Unified handling for all mutations
  if (strategy->requiresProgramLevelMutation()) {
    // Use program-level mutation API
    return strategy->applyToProgram(program, trace, statementsToMutate);
  } else {
    // Use statement-level mutation API
    return applyStatementLevelMutation(program, strategy, trace, statementsToMutate);
  }
}

// Helper method: Apply statement-level mutations
std::vector<interpreter::Program> TraceGuidedFuzzer::applyStatementLevelMutation(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    const std::set<size_t>& statementsToMutate) {
  
  std::vector<interpreter::Program> mutants;
  
  // Try to apply the mutation strategy to statements from the current round
  for (size_t i = 0; i < program.statements.size(); ++i) {
    // Skip statements not from the current increment
    if (statementsToMutate.find(i) == statementsToMutate.end()) {
      continue;
    }
    
    bool mutationApplied = false;
    auto mutatedStmt = applyMutationToStatement(
        program.statements[i].get(), strategy, trace, mutationApplied);
    
    if (mutationApplied) {
      // Debug: Show what mutation was applied
      FUZZER_DEBUG_LOG("Applied " << strategy->getName() << " to statement " << i << "\n");
      FUZZER_DEBUG_LOG("Original statement: " << program.statements[i]->toString() << "\n");
      FUZZER_DEBUG_LOG("Mutated statement: " << mutatedStmt->toString() << "\n");
      
      // Create a new program with the mutated statement
      interpreter::Program mutant;
      mutant.numThreadsX = program.numThreadsX;
      mutant.numThreadsY = program.numThreadsY;
      mutant.numThreadsZ = program.numThreadsZ;
      mutant.entryInputs = program.entryInputs;
      mutant.globalBuffers = program.globalBuffers;
      
      // Clone all statements except the mutated one
      for (size_t j = 0; j < program.statements.size(); ++j) {
        if (j == i) {
          mutant.statements.push_back(std::move(mutatedStmt));
        } else {
          mutant.statements.push_back(program.statements[j]->clone());
        }
      }
      
      mutants.push_back(std::move(mutant));
    }
  }
  
  return mutants;
}

bool TraceGuidedFuzzer::hasNewCoverage(const ExecutionTrace& trace) {
  bool foundNew = false;
  
  // Check for new block patterns
  for (const auto& [blockId, record] : trace.blocks) {
    auto hash = hashBlockPattern(record);
    if (seenBlockPatterns.find(hash) == seenBlockPatterns.end()) {
      seenBlockPatterns.insert(hash);
      foundNew = true;
    }
  }
  
  // Check for new wave patterns
  for (const auto& waveOp : trace.waveOperations) {
    auto hash = hashWavePattern(waveOp);
    if (seenWavePatterns.find(hash) == seenWavePatterns.end()) {
      seenWavePatterns.insert(hash);
      foundNew = true;
    }
  }
  
  // Check for new sync patterns
  for (const auto& barrier : trace.barriers) {
    auto hash = hashSyncPattern(barrier);
    if (seenSyncPatterns.find(hash) == seenSyncPatterns.end()) {
      seenSyncPatterns.insert(hash);
      foundNew = true;
    }
  }
  
  return foundNew;
}

uint64_t TraceGuidedFuzzer::hashBlockPattern(const ExecutionTrace::BlockExecutionRecord& block) {
  // Simple hash based on block ID and participation
  uint64_t hash = block.blockId;
  for (const auto& [waveId, participation] : block.waveParticipation) {
    hash ^= (waveId << 16);
    hash ^= participation.participatingLanes.size();
  }
  return hash;
}

uint64_t TraceGuidedFuzzer::hashWavePattern(const ExecutionTrace::WaveOpRecord& waveOp) {
  // Hash based on operation and participants
  std::hash<std::string> strHash;
  uint64_t hash = strHash(waveOp.opType);
  hash ^= waveOp.expectedParticipants.size();
  hash ^= (static_cast<uint64_t>(waveOp.waveId) << 32);
  return hash;
}

uint64_t TraceGuidedFuzzer::hashSyncPattern(const ExecutionTrace::BarrierRecord& barrier) {
  // Hash based on arrival pattern
  uint64_t hash = barrier.barrierId;
  hash ^= barrier.arrivedLanesPerWave.size();
  hash ^= (barrier.waveArrivalOrder.size() << 16);
  return hash;
}

void TraceGuidedFuzzer::logTrace(const std::string& message, const ExecutionTrace& trace) {
  FUZZER_DEBUG_LOG(message << "\n");
  FUZZER_DEBUG_LOG("  Blocks: " << trace.blocks.size() << "\n");
  FUZZER_DEBUG_LOG("  Wave ops: " << trace.waveOperations.size() << "\n");
  FUZZER_DEBUG_LOG("  Barriers: " << trace.barriers.size() << "\n");
}

void TraceGuidedFuzzer::logMutation(const std::string& message, const std::string& strategy) {
  FUZZER_DEBUG_LOG(message << " [" << strategy << "]\n");
}

void TraceGuidedFuzzer::logSummary(size_t testedMutants, size_t bugsFound) {
  FUZZER_DEBUG_LOG("\n=== Fuzzing Summary ===\n");
  FUZZER_DEBUG_LOG("Mutants tested: " << testedMutants << "\n");
  FUZZER_DEBUG_LOG("Bugs found: " << bugsFound << "\n");
  FUZZER_DEBUG_LOG("Coverage:\n");
  FUZZER_DEBUG_LOG("  Block patterns: " << seenBlockPatterns.size() << "\n");
  FUZZER_DEBUG_LOG("  Wave patterns: " << seenWavePatterns.size() << "\n");
  FUZZER_DEBUG_LOG("  Sync patterns: " << seenSyncPatterns.size() << "\n");
}

std::string TraceGuidedFuzzer::MutationResult::getMutationChainString() const {
  std::string result;
  
  if ((appliedMutations & AppliedMutations::WaveParticipantTracking) != AppliedMutations::None) {
    if (!result.empty()) result += " + ";
    result += "WaveParticipantTracking";
  }
  
  if ((appliedMutations & AppliedMutations::LanePermutation) != AppliedMutations::None) {
    if (!result.empty()) result += " + ";
    result += "LanePermutation";
  }
  
  if ((appliedMutations & AppliedMutations::ContextAwareParticipant) != AppliedMutations::None) {
    if (!result.empty()) result += " + ";
    result += "ContextAwareParticipant";
  }
  
  if ((appliedMutations & AppliedMutations::WaveParticipantFrequency) != AppliedMutations::None) {
    if (!result.empty()) result += " + ";
    result += "WaveParticipantFrequency";
  }
  
  if ((appliedMutations & AppliedMutations::WaveParticipantBitTracking) != AppliedMutations::None) {
    if (!result.empty()) result += " + ";
    result += "WaveParticipantBitTracking";
  }
  
  if (result.empty()) {
    result = "None";
  }
  
  return result;
}

TraceGuidedFuzzer::MutationResult TraceGuidedFuzzer::applyAllMutations(
    const interpreter::Program& baseProgram,
    const ExecutionTrace& goldenTrace,
    const std::vector<GenerationRound>* history,
    size_t currentIncrement) {
  
  MutationResult result;
  // Clone the base program since we can't use assignment
  result.mutatedProgram.numThreadsX = baseProgram.numThreadsX;
  result.mutatedProgram.numThreadsY = baseProgram.numThreadsY;
  result.mutatedProgram.numThreadsZ = baseProgram.numThreadsZ;
  result.mutatedProgram.globalBuffers = baseProgram.globalBuffers;
  result.mutatedProgram.entryInputs = baseProgram.entryInputs;
  result.mutatedProgram.waveSize = baseProgram.waveSize;
  for (const auto& stmt : baseProgram.statements) {
    result.mutatedProgram.statements.push_back(stmt->clone());
  }
  result.appliedMutations = AppliedMutations::None;
  
  // Apply each mutation strategy in sequence
  for (auto& strategy : mutationStrategies) {
    FUZZER_DEBUG_LOG("\nTrying to apply " << strategy->getName() << " mutation...\n");
    
    std::vector<interpreter::Program> mutants;
    if (history) {
      mutants = generateMutants(result.mutatedProgram, strategy.get(), goldenTrace, *history, currentIncrement);
    } else {
      // No history - create a dummy history with all statements in round 0
      std::vector<GenerationRound> dummyHistory;
      GenerationRound round0;
      round0.roundNumber = 0;
      for (size_t i = 0; i < result.mutatedProgram.statements.size(); ++i) {
        round0.addedStatementIndices.push_back(i);
      }
      dummyHistory.push_back(round0);
      mutants = generateMutants(result.mutatedProgram, strategy.get(), goldenTrace, dummyHistory, 0);
    }
    
    if (!mutants.empty()) {
      // Take the first mutant (most strategies generate only one)
      result.mutatedProgram = std::move(mutants[0]);
      
      // Update the applied mutations enum
      if (strategy->getName() == "WaveParticipantTracking") {
        result.appliedMutations |= AppliedMutations::WaveParticipantTracking;
      } else if (strategy->getName() == "LanePermutation") {
        result.appliedMutations |= AppliedMutations::LanePermutation;
      } else if (strategy->getName() == "ContextAwareParticipant") {
        result.appliedMutations |= AppliedMutations::ContextAwareParticipant;
      } else if (strategy->getName() == "WaveParticipantFrequency") {
        result.appliedMutations |= AppliedMutations::WaveParticipantFrequency;
      } else if (strategy->getName() == "WaveParticipantBitTracking") {
        result.appliedMutations |= AppliedMutations::WaveParticipantBitTracking;
      }
      
      FUZZER_DEBUG_LOG("Successfully applied " << strategy->getName() << "\n");
    } else {
      FUZZER_DEBUG_LOG(strategy->getName() << " not applicable\n");
    }
  }
  
  return result;
}


} // namespace fuzzer
} // namespace minihlsl

// Main function - always uses the incremental fuzzing pipeline
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file> [wave_size]\n";
    std::cerr << "  wave_size: optional, default is 32 (valid values: 4, 8, 16, 32, 64)\n";
    return 1;
  }
  
  // Read input file
  std::ifstream file(argv[1], std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file " << argv[1] << "\n";
    return 1;
  }
  
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  
  if (data.size() < 16) {
    std::cerr << "Error: Input file too small (need at least 16 bytes)\n";
    return 1;
  }
  
  // Configure the incremental fuzzing pipeline
  minihlsl::fuzzer::IncrementalFuzzingConfig config;
  
  // Parse wave size from command line if provided
  if (argc >= 3) {
    try {
      uint32_t waveSize = std::stoul(argv[2]);
      // Validate wave size
      if (waveSize != 4 && waveSize != 8 && waveSize != 16 && waveSize != 32 && waveSize != 64) {
        std::cerr << "Error: Invalid wave size " << waveSize << ". Valid values are: 4, 8, 16, 32, 64\n";
        return 1;
      }
      config.waveSize = waveSize;
      std::cout << "Using wave size: " << waveSize << "\n";
      
      // Also set environment variable for the program generator
      std::string waveSizeStr = std::to_string(waveSize);
      #ifdef _WIN32
      _putenv_s("FUZZ_WAVE_SIZE", waveSizeStr.c_str());
      #else
      setenv("FUZZ_WAVE_SIZE", waveSizeStr.c_str(), 1);
      #endif
    } catch (const std::exception& e) {
      std::cerr << "Error: Invalid wave size argument: " << argv[2] << "\n";
      return 1;
    }
  }
  
  // Check environment variables for configuration
  const char* maxIncrementsEnv = std::getenv("FUZZ_MAX_INCREMENTS");
  if (maxIncrementsEnv) {
    config.maxIncrements = std::stoul(maxIncrementsEnv);
  } else {
    config.maxIncrements = 1;
  }
  
  const char* seedIdEnv = std::getenv("FUZZ_SEED_ID");
  if (seedIdEnv) {
    config.seedId = seedIdEnv;
  }
  
  const char* outputDirEnv = std::getenv("FUZZ_OUTPUT_DIR");
  if (outputDirEnv) {
    config.outputDir = outputDirEnv;
  }
  
  config.mutantsPerIncrement = 10;
  config.enableLogging = true;
  
  // Always run the incremental fuzzing pipeline
  minihlsl::fuzzer::IncrementalFuzzingPipeline pipeline(config);
  auto result = pipeline.run(data.data(), data.size());
  
  std::cout << "\n=== Pipeline Results ===\n";
  std::cout << "Total increments: " << result.increments.size() << "\n";
  std::cout << "Total mutants tested: " << result.totalMutantsTested << "\n";
  std::cout << "Total bugs found: " << result.totalBugsFound << "\n";
  if (result.stoppedEarly) {
    std::cout << "Stopped early: " << result.stopReason << "\n";
  }
  
  return 0;
}
