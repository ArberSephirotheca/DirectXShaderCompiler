#include "MiniHLSLValidator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void testValidCases() {
    std::cout << "=== Testing Valid MiniHLSL Cases ===" << std::endl;
    
    std::string validSource = readFile("test_cases/valid_minihlsl.hlsl");
    if (validSource.empty()) {
        std::cout << "Could not read valid test cases" << std::endl;
        return;
    }
    
    minihlsl::MiniHLSLValidator validator;
    auto result = validator.validateSource(validSource);
    
    std::cout << "Valid test cases validation: " << (result.isValid ? "PASS" : "FAIL") << std::endl;
    
    if (!result.isValid) {
        std::cout << "Unexpected validation errors:" << std::endl;
        for (const auto& error : result.errorMessages) {
            std::cout << "  - " << error << std::endl;
        }
    }
}

void testInvalidCases() {
    std::cout << "\n=== Testing Invalid MiniHLSL Cases ===" << std::endl;
    
    std::string invalidSource = readFile("test_cases/invalid_minihlsl.hlsl");
    if (invalidSource.empty()) {
        std::cout << "Could not read invalid test cases" << std::endl;
        return;
    }
    
    minihlsl::MiniHLSLValidator validator;
    auto result = validator.validateSource(invalidSource);
    
    std::cout << "Invalid test cases validation: " << (!result.isValid ? "PASS" : "FAIL") << std::endl;
    
    if (result.isValid) {
        std::cout << "ERROR: Invalid cases should have failed validation!" << std::endl;
    } else {
        std::cout << "Detected validation errors (expected):" << std::endl;
        for (size_t i = 0; i < result.errorMessages.size() && i < 10; ++i) {
            std::cout << "  - " << result.errorMessages[i] << std::endl;
        }
        if (result.errorMessages.size() > 10) {
            std::cout << "  ... and " << (result.errorMessages.size() - 10) << " more errors" << std::endl;
        }
    }
}

void testOrderIndependentGeneration() {
    std::cout << "\n=== Testing Order-Independent Generation ===" << std::endl;
    
    // Test with a simple base program
    std::string baseProgram = R"(
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    float sum = WaveActiveSum(float(lane));
}
)";
    
    auto variants = minihlsl::generateOrderIndependentVariants(baseProgram);
    std::cout << "Generated " << variants.size() << " order-independent variants" << std::endl;
    
    minihlsl::MiniHLSLValidator validator;
    int validCount = 0;
    
    for (size_t i = 0; i < variants.size(); ++i) {
        auto result = validator.validateSource(variants[i]);
        if (result.isValid) {
            validCount++;
        }
        std::cout << "Variant " << (i + 1) << ": " << (result.isValid ? "VALID" : "INVALID") << std::endl;
    }
    
    std::cout << "Valid variants: " << validCount << "/" << variants.size() << std::endl;
}

void testSpecificConstructs() {
    std::cout << "\n=== Testing Specific Constructs ===" << std::endl;
    
    minihlsl::MiniHLSLValidator validator;
    
    // Test forbidden keywords
    std::vector<std::string> testCases = {
        "WavePrefixSum",     // Should be forbidden
        "WaveActiveSum",     // Should be allowed
        "for",               // Should be forbidden
        "if",                // Should be allowed (context-dependent)
        "WaveReadLaneAt",    // Should be forbidden
        "WaveGetLaneIndex"   // Should be allowed
    };
    
    for (const auto& construct : testCases) {
        bool allowed = validator.isAllowedConstruct(construct);
        std::cout << construct << ": " << (allowed ? "ALLOWED" : "FORBIDDEN") << std::endl;
    }
}

void testUtilityFunctions() {
    std::cout << "\n=== Testing Utility Functions ===" << std::endl;
    
    // Test commutative operations
    std::vector<std::string> ops = {"+", "-", "*", "/", "==", "!=", "&&", "||"};
    
    std::cout << "Commutative operations:" << std::endl;
    for (const auto& op : ops) {
        bool commutative = minihlsl::isCommutativeOperation(op);
        std::cout << "  " << op << ": " << (commutative ? "YES" : "NO") << std::endl;
    }
    
    std::cout << "\nAssociative operations:" << std::endl;
    for (const auto& op : ops) {
        bool associative = minihlsl::isAssociativeOperation(op);
        std::cout << "  " << op << ": " << (associative ? "YES" : "NO") << std::endl;
    }
}

int main() {
    std::cout << "MiniHLSL Validator Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        testValidCases();
        testInvalidCases();
        testOrderIndependentGeneration();
        testSpecificConstructs();
        testUtilityFunctions();
        
        std::cout << "\n=== Test Suite Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}