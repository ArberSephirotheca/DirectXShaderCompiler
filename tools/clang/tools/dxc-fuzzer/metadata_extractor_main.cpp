#include <iostream>
#include <fstream>
#include <filesystem>
#include <regex>
#include <thread>
#include <mutex>
#include <queue>
#include <map>
#include <json/json.h>

#include "MiniHLSLInterpreter.h"
#include "MiniHLSLInterpreterFuzzer.h"

namespace fs = std::filesystem;

// Metadata capturing version of HLSLProgramGenerator
class MetadataCapturingGenerator : public minihlsl::fuzzer::HLSLProgramGenerator {
public:
    struct GenerationMetadata {
        // Core identification
        std::string seedId;
        uint32_t increment;
        std::string mutation;
        
        // Structural information
        struct Structure {
            uint32_t nestingDepth = 0;
            std::vector<std::string> controlFlowStack;
            uint32_t loopCount = 0;
            uint32_t conditionalCount = 0;
            uint32_t switchCount = 0;
            bool hasElse = false;
            bool hasBreakContinue = false;
        } structure;
        
        // Wave operation information
        struct WaveOpInfo {
            std::string opType;
            std::string enclosingBlock;
            uint32_t nestingLevel;
            bool isInLoop;
            bool isInConditional;
            std::string participantPattern;
        };
        std::vector<WaveOpInfo> waveOps;
        
        // Pattern classification
        std::string patternId;
        std::string patternCategory;
        std::string patternDescription;
        uint32_t complexityScore;
    };
    
    MetadataCapturingGenerator() : currentNestingLevel(0) {}
    
    // Override generation methods to capture metadata
    void generateIfStatement(FuzzedDataProvider& provider) override {
        metadata.structure.conditionalCount++;
        controlFlowStack.push_back("if");
        currentNestingLevel++;
        metadata.structure.nestingDepth = std::max(metadata.structure.nestingDepth, currentNestingLevel);
        
        // Check if it has else
        bool hasElse = provider.ConsumeBool();
        if (hasElse) {
            metadata.structure.hasElse = true;
        }
        
        // Call parent implementation
        HLSLProgramGenerator::generateIfStatement(provider);
        
        currentNestingLevel--;
        controlFlowStack.pop_back();
    }
    
    void generateForLoop(FuzzedDataProvider& provider) override {
        metadata.structure.loopCount++;
        controlFlowStack.push_back("for");
        currentNestingLevel++;
        metadata.structure.nestingDepth = std::max(metadata.structure.nestingDepth, currentNestingLevel);
        
        // Call parent implementation
        HLSLProgramGenerator::generateForLoop(provider);
        
        currentNestingLevel--;
        controlFlowStack.pop_back();
    }
    
    void generateWhileLoop(FuzzedDataProvider& provider) override {
        metadata.structure.loopCount++;
        controlFlowStack.push_back("while");
        currentNestingLevel++;
        metadata.structure.nestingDepth = std::max(metadata.structure.nestingDepth, currentNestingLevel);
        
        // Call parent implementation
        HLSLProgramGenerator::generateWhileLoop(provider);
        
        currentNestingLevel--;
        controlFlowStack.pop_back();
    }
    
    void generateWaveOperation(FuzzedDataProvider& provider, const std::string& opType) override {
        WaveOpInfo waveOp;
        waveOp.opType = opType;
        waveOp.enclosingBlock = getCurrentBlockName();
        waveOp.nestingLevel = currentNestingLevel;
        waveOp.isInLoop = isInLoop();
        waveOp.isInConditional = isInConditional();
        
        // Capture participant pattern if available
        // This is simplified - in real implementation would track actual pattern used
        waveOp.participantPattern = "unknown";
        
        metadata.waveOps.push_back(waveOp);
        
        // Call parent implementation
        HLSLProgramGenerator::generateWaveOperation(provider, opType);
    }
    
    // Generate program with metadata capture
    interpreter::Program generateWithMetadata(const std::vector<uint8_t>& seedData, 
                                            uint32_t increment,
                                            uint32_t waveSize = 32,
                                            uint32_t numThreads = 64) {
        // Reset metadata
        metadata = GenerationMetadata();
        metadata.increment = increment;
        controlFlowStack.clear();
        currentNestingLevel = 0;
        
        // Create FuzzedDataProvider
        FuzzedDataProvider provider(seedData.data(), seedData.size());
        
        // Set generation parameters
        setWaveSize(waveSize);
        setNumThreads(numThreads);
        
        // Generate base program
        auto program = generateBaseProgram(provider, increment);
        
        // Classify pattern based on captured metadata
        classifyPattern();
        
        return program;
    }
    
    // Apply mutations and update metadata
    void applyMutation(const std::string& mutationType, interpreter::Program& program) {
        metadata.mutation = mutationType;
        
        // Apply the mutation (simplified - would call actual mutation methods)
        if (mutationType == "WaveParticipantTracking") {
            // Apply WaveParticipantTracking mutation
            applyWaveParticipantTrackingMutation(program);
        } else if (mutationType == "LanePermutation") {
            // Apply LanePermutation mutation
            applyLanePermutationMutation(program);
        } else if (mutationType == "ContextAwareParticipant") {
            // Apply ContextAwareParticipant mutation
            applyContextAwareMutation(program);
        }
    }
    
    const GenerationMetadata& getMetadata() const { return metadata; }
    
private:
    GenerationMetadata metadata;
    std::vector<std::string> controlFlowStack;
    uint32_t currentNestingLevel;
    
    std::string getCurrentBlockName() {
        std::stringstream ss;
        for (size_t i = 0; i < controlFlowStack.size(); i++) {
            if (i > 0) ss << "_";
            ss << controlFlowStack[i] << "_" << i;
        }
        return ss.str();
    }
    
    bool isInLoop() {
        for (const auto& cf : controlFlowStack) {
            if (cf == "for" || cf == "while" || cf == "do") {
                return true;
            }
        }
        return false;
    }
    
    bool isInConditional() {
        for (const auto& cf : controlFlowStack) {
            if (cf == "if" || cf == "switch") {
                return true;
            }
        }
        return false;
    }
    
    void classifyPattern() {
        // Pattern classification based on metadata
        if (metadata.structure.nestingDepth == 0) {
            metadata.patternId = "P01";
            metadata.patternCategory = "simple";
            metadata.patternDescription = "Simple wave operation, no control flow";
            metadata.complexityScore = 1;
        }
        else if (metadata.structure.nestingDepth == 1) {
            if (metadata.structure.loopCount > 0) {
                metadata.patternId = "P21";
                metadata.patternCategory = "single_loop";
                metadata.patternDescription = "Wave operation in single loop";
                metadata.complexityScore = 2;
            } else if (metadata.structure.conditionalCount > 0) {
                metadata.patternId = metadata.structure.hasElse ? "P22" : "P23";
                metadata.patternCategory = "single_conditional";
                metadata.patternDescription = metadata.structure.hasElse ? 
                    "Wave operation in if-else" : "Wave operation in if (no else)";
                metadata.complexityScore = 2;
            }
        }
        else if (metadata.structure.nestingDepth >= 2) {
            // Analyze nesting pattern
            if (controlFlowStack.size() >= 2) {
                std::string pattern = controlFlowStack[0] + "_" + controlFlowStack[1];
                if (pattern == "for_if") {
                    metadata.patternId = "P41";
                    metadata.patternCategory = "nested_for_if";
                    metadata.patternDescription = "If statement inside for loop with wave op";
                    metadata.complexityScore = 3;
                } else if (pattern == "if_for") {
                    metadata.patternId = "P42";
                    metadata.patternCategory = "nested_if_for";
                    metadata.patternDescription = "For loop inside if statement with wave op";
                    metadata.complexityScore = 3;
                } else if (pattern == "for_for") {
                    metadata.patternId = "P43";
                    metadata.patternCategory = "nested_loops";
                    metadata.patternDescription = "Nested loops with wave op";
                    metadata.complexityScore = 4;
                }
            }
        }
        
        // Special patterns based on mutation
        if (metadata.mutation == "ContextAwareParticipant") {
            metadata.patternId = "P71";
            metadata.patternCategory = "context_aware";
            metadata.patternDescription = "Context-aware participant mutation";
            metadata.complexityScore = 5;
        }
        
        // Default for unclassified
        if (metadata.patternId.empty()) {
            metadata.patternId = "P99";
            metadata.patternCategory = "complex";
            metadata.patternDescription = "Complex unclassified pattern";
            metadata.complexityScore = 5;
        }
    }
};

// Test information parser
struct TestInfo {
    std::string seedId;
    uint32_t increment;
    std::string mutation;
    std::string testPath;
    
    static TestInfo parseTestName(const std::string& testPath) {
        TestInfo info;
        info.testPath = testPath;
        
        // Parse: program_1735064431123456789_1_increment_0_WaveParticipantTracking.test
        fs::path p(testPath);
        std::string filename = p.filename().string();
        
        std::regex pattern(R"(program_(.+?)_increment_(\d+)_(\w+)\.test)");
        std::smatch match;
        
        if (std::regex_match(filename, match, pattern)) {
            info.seedId = match[1];
            info.increment = std::stoul(match[2]);
            info.mutation = match[3];
        }
        
        return info;
    }
};

// Metadata extractor
class MetadataExtractor {
public:
    void extractMetadata(const std::string& seedDir, 
                        const std::string& testDir,
                        const std::string& outputDir,
                        uint32_t waveSize = 32,
                        uint32_t numThreads = 64) {
        // Create output directory
        fs::create_directories(outputDir);
        
        // Collect all seed files
        std::vector<fs::path> seedFiles;
        for (const auto& entry : fs::directory_iterator(seedDir)) {
            if (entry.path().extension() == ".bin") {
                seedFiles.push_back(entry.path());
            }
        }
        
        std::cout << "Found " << seedFiles.size() << " seed files\n";
        
        // Process each seed
        size_t processed = 0;
        for (const auto& seedFile : seedFiles) {
            processSeed(seedFile, testDir, outputDir, waveSize, numThreads);
            processed++;
            
            if (processed % 10 == 0) {
                std::cout << "Processed " << processed << "/" << seedFiles.size() << " seeds\n";
            }
        }
        
        std::cout << "Metadata extraction complete!\n";
    }
    
private:
    void processSeed(const fs::path& seedFile,
                    const std::string& testDir,
                    const std::string& outputDir,
                    uint32_t waveSize,
                    uint32_t numThreads) {
        // Extract seed ID from filename
        std::string seedFilename = seedFile.filename().string();
        std::string seedId = seedFilename.substr(5, seedFilename.length() - 9); // Remove "seed_" and ".bin"
        
        // Load seed data
        std::ifstream file(seedFile, std::ios::binary);
        std::vector<uint8_t> seedData((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        
        // Find all tests for this seed
        std::vector<TestInfo> tests;
        for (const auto& entry : fs::directory_iterator(testDir)) {
            if (entry.path().extension() == ".test") {
                TestInfo info = TestInfo::parseTestName(entry.path().string());
                if (info.seedId == seedId) {
                    tests.push_back(info);
                }
            }
        }
        
        // Process each test
        for (const auto& test : tests) {
            try {
                // Create generator
                MetadataCapturingGenerator generator;
                
                // Generate program with metadata
                auto program = generator.generateWithMetadata(seedData, test.increment, 
                                                             waveSize, numThreads);
                
                // Apply mutation
                generator.applyMutation(test.mutation, program);
                
                // Get metadata
                auto metadata = generator.getMetadata();
                metadata.seedId = seedId;
                
                // Save metadata as JSON
                saveMetadata(test, metadata, outputDir);
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << test.testPath << ": " << e.what() << "\n";
            }
        }
    }
    
    void saveMetadata(const TestInfo& test, 
                     const MetadataCapturingGenerator::GenerationMetadata& metadata,
                     const std::string& outputDir) {
        // Create JSON output
        Json::Value root;
        
        // Test identification
        root["test_name"] = fs::path(test.testPath).filename().string();
        root["seed_id"] = metadata.seedId;
        root["increment"] = metadata.increment;
        root["mutation"] = metadata.mutation;
        
        // Structural information
        Json::Value structure;
        structure["nesting_depth"] = metadata.structure.nestingDepth;
        structure["control_flow"] = Json::arrayValue;
        for (const auto& cf : metadata.structure.controlFlowStack) {
            structure["control_flow"].append(cf);
        }
        structure["loop_count"] = metadata.structure.loopCount;
        structure["conditional_count"] = metadata.structure.conditionalCount;
        structure["has_else"] = metadata.structure.hasElse;
        root["structure"] = structure;
        
        // Wave operations
        Json::Value waveOps(Json::arrayValue);
        for (const auto& op : metadata.waveOps) {
            Json::Value opJson;
            opJson["type"] = op.opType;
            opJson["enclosing_block"] = op.enclosingBlock;
            opJson["nesting_level"] = op.nestingLevel;
            opJson["in_loop"] = op.isInLoop;
            opJson["in_conditional"] = op.isInConditional;
            opJson["participant_pattern"] = op.participantPattern;
            waveOps.append(opJson);
        }
        root["wave_operations"] = waveOps;
        
        // Pattern classification
        root["pattern"]["id"] = metadata.patternId;
        root["pattern"]["category"] = metadata.patternCategory;
        root["pattern"]["description"] = metadata.patternDescription;
        root["pattern"]["complexity"] = metadata.complexityScore;
        
        // Write to file
        std::string outputPath = outputDir + "/" + 
                                fs::path(test.testPath).filename().string() + ".meta.json";
        
        std::ofstream outFile(outputPath);
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(root, &outFile);
        outFile << std::endl;
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <seed_dir> <test_dir> <output_dir> [wave_size] [num_threads]\n";
        std::cerr << "\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  seed_dir    - Directory containing seed_*.bin files\n";
        std::cerr << "  test_dir    - Directory containing *.test files\n";
        std::cerr << "  output_dir  - Directory to write metadata JSON files\n";
        std::cerr << "  wave_size   - Wave size (default: 32)\n";
        std::cerr << "  num_threads - Number of threads (default: 64)\n";
        return 1;
    }
    
    std::string seedDir = argv[1];
    std::string testDir = argv[2];
    std::string outputDir = argv[3];
    uint32_t waveSize = (argc > 4) ? std::stoul(argv[4]) : 32;
    uint32_t numThreads = (argc > 5) ? std::stoul(argv[5]) : 64;
    
    std::cout << "=== Metadata Extractor ===\n";
    std::cout << "Seed directory: " << seedDir << "\n";
    std::cout << "Test directory: " << testDir << "\n";
    std::cout << "Output directory: " << outputDir << "\n";
    std::cout << "Wave size: " << waveSize << "\n";
    std::cout << "Num threads: " << numThreads << "\n\n";
    
    try {
        MetadataExtractor extractor;
        extractor.extractMetadata(seedDir, testDir, outputDir, waveSize, numThreads);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}