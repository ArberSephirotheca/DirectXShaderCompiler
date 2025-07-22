#include "MiniHLSLValidator.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

using namespace minihlsl;
namespace fs = std::filesystem;

struct TestResult {
    std::string filename;
    bool passed;
    std::vector<std::string> errors;
};

std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open: " + filepath);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

bool should_pass(const std::string& filename) {
    // Files with "valid" in name should pass, "invalid" should fail
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Check for "invalid" first (more specific)
    if (lower.find("invalid") != std::string::npos) {
        return false;  // Files with "invalid" should fail validation
    }
    
    // Then check for "valid"
    if (lower.find("valid") != std::string::npos) {
        return true;   // Files with "valid" should pass validation
    }
    
    // Default: assume should pass if neither "valid" nor "invalid" in name
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_directory>" << std::endl;
        return 1;
    }
    
    std::string test_dir = argv[1];
    
    if (!fs::exists(test_dir) || !fs::is_directory(test_dir)) {
        std::cerr << "Error: " << test_dir << " is not a valid directory" << std::endl;
        return 1;
    }
    
    std::cout << "MiniHLSL Validator Test Runner" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Test directory: " << test_dir << std::endl << std::endl;
    
    auto validator = ValidatorFactory::create_validator();
    std::vector<TestResult> results;
    
    // Find all .hlsl files
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".hlsl") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();
            
            std::cout << "Testing: " << filename << " ... ";
            
            try {
                std::string source = read_file(filepath);
                auto result = validator->validate_source_with_full_ast(source, filename);
                
                bool expected_pass = should_pass(filename);
                bool actually_passed = result.is_ok();
                bool test_passed = (expected_pass == actually_passed);
                
                TestResult test_result;
                test_result.filename = filename;
                test_result.passed = test_passed;
                
                if (test_passed) {
                    std::cout << "PASS";
                    if (expected_pass) {
                        std::cout << " (valid)";
                    } else {
                        std::cout << " (correctly rejected)";
                    }
                } else {
                    std::cout << "FAIL";
                    if (expected_pass && !actually_passed) {
                        std::cout << " (should pass but failed)";
                        const auto& errors = result.unwrap_err();
                        for (const auto& error : errors) {
                            test_result.errors.push_back(error.message);
                        }
                    } else if (!expected_pass && actually_passed) {
                        std::cout << " (should fail but passed)";
                    }
                }
                std::cout << std::endl;
                
                results.push_back(test_result);
                
            } catch (const std::exception& e) {
                std::cout << "ERROR: " << e.what() << std::endl;
                TestResult test_result;
                test_result.filename = filename;
                test_result.passed = false;
                test_result.errors.push_back(e.what());
                results.push_back(test_result);
            }
        }
    }
    
    // Print summary report
    std::cout << std::endl;
    std::cout << "========== TEST REPORT ==========" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& result : results) {
        if (result.passed) {
            passed++;
        } else {
            failed++;
            std::cout << "FAILED: " << result.filename << std::endl;
            for (const auto& error : result.errors) {
                std::cout << "  - " << error << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
    std::cout << "Total tests: " << results.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success rate: " << (results.empty() ? 0 : (100 * passed / results.size())) << "%" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return failed > 0 ? 1 : 0;
}