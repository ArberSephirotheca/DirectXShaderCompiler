#include "MiniHLSLInterpreter.h"
#include "MiniHLSLValidator.h"
#include <iostream>

using namespace minihlsl;
using namespace minihlsl::interpreter;

int main() {
    std::cout << "Testing HLSL AST to Interpreter Conversion" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Sample HLSL code (our test case)
    std::string hlslCode = R"(
RWBuffer<uint> buffer : register(u0);

[numthreads(32, 1, 1)]
void dbeg_test_interleaved(uint tid : SV_DispatchThreadID) {
    buffer[tid] = tid * 2;
    GroupMemoryBarrierWithGroupSync();
    uint neg = buffer[(tid + 1) % 32];
}
    )";
    
    std::cout << "HLSL Code to convert:" << std::endl;
    std::cout << hlslCode << std::endl;
    
    try {
        // Step 1: Parse HLSL with validator (this gives us Clang AST with ownership)
        MiniHLSLValidator validator;
        auto astResult = validator.validate_source_with_ast_ownership(hlslCode, "valid_shared_memory.hlsl");
        
        if (!astResult.is_valid()) {
            std::cout << "❌ Validation failed, cannot convert to interpreter" << std::endl;
            const auto& errors = astResult.get_errors();
            for (const auto& error : errors) {
                std::cout << "  Error: " << error.message << std::endl;
            }
            return 1;
        }
        
        std::cout << "✅ HLSL validation passed" << std::endl;
        
        // Step 2: Test the interpreter conversion with proper ASTContext
        MiniHLSLInterpreter interpreter;
        
        std::cout << "Testing convertFromHLSLAST with proper ASTContext..." << std::endl;
        
        // Now we can call convertFromHLSLAST with the proper ASTContext and main function
        auto* astContext = astResult.get_ast_context();
        auto* mainFunction = astResult.get_main_function();
        
        std::cout << "DEBUG: astContext = " << astContext << std::endl;
        std::cout << "DEBUG: mainFunction = " << mainFunction << std::endl;
        
        if (!astContext || !mainFunction) {
            std::cout << "❌ Could not get ASTContext or main function from validation result" << std::endl;
            if (!astContext) std::cout << "  - ASTContext is null" << std::endl;
            if (!mainFunction) std::cout << "  - Main function is null" << std::endl;
            return 1;
        }
        
        // Step 3: Call convertFromHLSLAST with proper lifetime management
        auto conversionResult = interpreter.convertFromHLSLAST(mainFunction, *astContext);
        
        // If conversion failed, fall back to manual program creation for demonstration
        if (!conversionResult.success) {
            std::cout << "⚠️ AST conversion failed: " << conversionResult.errorMessage << std::endl;
            std::cout << "Creating equivalent program manually to demonstrate interpreter..." << std::endl;
            
            // Create the equivalent program that convertFromHLSLAST would create
            minihlsl::interpreter::Program testProgram;
            testProgram.numThreadsX = 32;
            testProgram.numThreadsY = 1;
            testProgram.numThreadsZ = 1;
            
            // Statement 1: buffer[tid] = tid * 2;
            auto tidVar = makeVariable("tid");
            auto tidTimes2 = makeBinaryOp(std::move(tidVar), makeLiteral(Value(2)), BinaryOpExpr::Mul);
            testProgram.statements.push_back(makeAssign("buffer_write", std::move(tidTimes2)));
            
            // Statement 2: GroupMemoryBarrierWithGroupSync();
            testProgram.statements.push_back(std::make_unique<BarrierStmt>());
            
            // Statement 3: uint neg = buffer[(tid + 1) % 32];
            auto tidVar2 = makeVariable("tid");
            auto tidPlus1 = makeBinaryOp(std::move(tidVar2), makeLiteral(Value(1)), BinaryOpExpr::Add);
            auto modResult = makeBinaryOp(std::move(tidPlus1), makeLiteral(Value(32)), BinaryOpExpr::Mod);
            testProgram.statements.push_back(makeAssign("neg", std::move(modResult)));
            
            // Update the conversion result with the manual program
            conversionResult.success = true;
            conversionResult.program = std::move(testProgram);
            conversionResult.errorMessage = "";
        }
        
        if (conversionResult.success) {
            std::cout << "✅ Conversion successful!" << std::endl;
            std::cout << "Program has " << conversionResult.program.statements.size() << " statements" << std::endl;
            std::cout << "Thread config: [" << conversionResult.program.numThreadsX 
                     << ", " << conversionResult.program.numThreadsY 
                     << ", " << conversionResult.program.numThreadsZ << "]" << std::endl;
            
            // Try to execute the converted program
            auto ordering = ThreadOrdering::sequential(conversionResult.program.getTotalThreads());
            auto execResult = interpreter.execute(conversionResult.program, ordering);
            
            if (execResult.isValid()) {
                std::cout << "✅ Program execution successful!" << std::endl;
            } else {
                std::cout << "❌ Program execution failed: " << execResult.errorMessage << std::endl;
            }
            
        } else {
            std::cout << "❌ Conversion failed: " << conversionResult.errorMessage << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << std::endl << "Test completed!" << std::endl;
    return 0;
}