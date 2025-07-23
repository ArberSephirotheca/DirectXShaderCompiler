#include "MiniHLSLInterpreter.h"
#include <iostream>
#include <iomanip>

using namespace minihlsl::interpreter;

// Example 1: Simple wave reduction
Program createWaveReductionTest() {
    Program program;
    program.numThreadsX = 32;
    
    // var laneId = WaveGetLaneIndex();
    program.statements.push_back(makeVarDecl("laneId", makeLaneIndex()));
    
    // var value = laneId * laneId;
    program.statements.push_back(makeVarDecl("value", 
        makeBinaryOp(makeVariable("laneId"), makeVariable("laneId"), BinaryOpExpr::Mul)));
    
    // var waveSum = WaveActiveSum(value);
    program.statements.push_back(makeVarDecl("waveSum", makeWaveSum(makeVariable("value"))));
    
    // return waveSum;
    program.statements.push_back(std::make_unique<ReturnStmt>(makeVariable("waveSum")));
    
    return program;
}

// Example 2: Deterministic control flow
Program createDeterministicBranchTest() {
    Program program;
    program.numThreadsX = 32;
    
    // var laneId = WaveGetLaneIndex();
    program.statements.push_back(makeVarDecl("laneId", makeLaneIndex()));
    
    // var result = 0;
    program.statements.push_back(makeVarDecl("result", makeLiteral(Value(0))));
    
    // if (laneId < 16) { result = 1; } else { result = 2; }
    std::vector<std::unique_ptr<Statement>> thenBlock;
    thenBlock.push_back(makeAssign("result", makeLiteral(Value(1))));
    
    std::vector<std::unique_ptr<Statement>> elseBlock;
    elseBlock.push_back(makeAssign("result", makeLiteral(Value(2))));
    
    program.statements.push_back(makeIf(
        makeBinaryOp(makeVariable("laneId"), makeLiteral(Value(16)), BinaryOpExpr::Lt),
        std::move(thenBlock),
        std::move(elseBlock)
    ));
    
    // var sum = WaveActiveSum(result);
    program.statements.push_back(makeVarDecl("sum", makeWaveSum(makeVariable("result"))));
    
    // return sum;
    program.statements.push_back(std::make_unique<ReturnStmt>(makeVariable("sum")));
    
    return program;
}

// Example 3: Shared memory with barriers
Program createSharedMemoryTest() {
    Program program;
    program.numThreadsX = 32;
    
    // var threadId = GetThreadIndex();
    program.statements.push_back(makeVarDecl("threadId", makeThreadIndex()));
    
    // g_shared[threadId] = threadId * 2; (each thread writes to its own location)
    program.statements.push_back(std::make_unique<SharedWriteStmt>(0, // simplified: use threadId as address
        makeBinaryOp(makeVariable("threadId"), makeLiteral(Value(2)), BinaryOpExpr::Mul)));
    
    // GroupMemoryBarrierWithGroupSync();
    program.statements.push_back(std::make_unique<BarrierStmt>());
    
    // var neighbor = g_shared[(threadId + 1) % 32];
    program.statements.push_back(makeVarDecl("neighbor", 
        std::make_unique<SharedReadExpr>(1))); // simplified
    
    // return neighbor;
    program.statements.push_back(std::make_unique<ReturnStmt>(makeVariable("neighbor")));
    
    return program;
}

// Example 4: Order-dependent program (should fail verification)
Program createOrderDependentTest() {
    Program program;
    program.numThreadsX = 4;
    
    // This program is intentionally order-dependent
    // All threads write to the same shared memory location without synchronization
    
    // var threadId = GetThreadIndex();
    program.statements.push_back(makeVarDecl("threadId", makeThreadIndex()));
    
    // g_shared[0] = threadId; (all threads write to same location - race condition!)
    program.statements.push_back(std::make_unique<SharedWriteStmt>(0, makeVariable("threadId")));
    
    // var value = g_shared[0];
    program.statements.push_back(makeVarDecl("value", std::make_unique<SharedReadExpr>(0)));
    
    // return value;
    program.statements.push_back(std::make_unique<ReturnStmt>(makeVariable("value")));
    
    return program;
}

void printExecutionResult(const ExecutionResult& result, const std::string& testName) {
    std::cout << "=== " << testName << " ===" << std::endl;
    
    if (!result.isValid()) {
        std::cout << "❌ Error: " << result.errorMessage << std::endl;
        return;
    }
    
    std::cout << "✅ Execution successful" << std::endl;
    
    // Print shared memory state
    if (!result.sharedMemoryState.empty()) {
        std::cout << "Shared Memory:" << std::endl;
        for (const auto& [addr, val] : result.sharedMemoryState) {
            std::cout << "  [" << addr << "] = " << val.toString() << std::endl;
        }
    }
    
    // Print return values (first few)
    if (!result.threadReturnValues.empty()) {
        std::cout << "Return Values (first 8): ";
        for (size_t i = 0; i < std::min(size_t(8), result.threadReturnValues.size()); ++i) {
            std::cout << result.threadReturnValues[i].toString();
            if (i < std::min(size_t(8), result.threadReturnValues.size()) - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
}

void printVerificationResult(const MiniHLSLInterpreter::VerificationResult& result, 
                           const std::string& testName) {
    std::cout << "=== Order Independence Verification: " << testName << " ===" << std::endl;
    
    if (result.isOrderIndependent) {
        std::cout << "✅ PASS: Program is order-independent!" << std::endl;
        std::cout << "Tested " << result.orderings.size() << " different thread orderings." << std::endl;
    } else {
        std::cout << "❌ FAIL: Program is order-dependent!" << std::endl;
        std::cout << result.divergenceReport << std::endl;
    }
    
    // Show the orderings tested
    std::cout << "Thread orderings tested:" << std::endl;
    for (const auto& ordering : result.orderings) {
        std::cout << "  - " << ordering.description << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "MiniHLSL Interpreter Test Suite" << std::endl;
    std::cout << "================================" << std::endl << std::endl;
    
    MiniHLSLInterpreter interpreter(42); // Fixed seed for reproducibility
    
    // Test 1: Wave Reduction
    {
        std::cout << "TEST 1: Wave Reduction (Order-Independent)" << std::endl;
        auto program = createWaveReductionTest();
        
        // Execute with sequential ordering first
        auto result = interpreter.execute(program, ThreadOrdering::sequential(32));
        printExecutionResult(result, "Sequential Execution");
        
        // Verify order independence
        auto verification = interpreter.verifyOrderIndependence(program, 5);
        printVerificationResult(verification, "Wave Reduction");
    }
    
    // Test 2: Deterministic Branching
    {
        std::cout << "TEST 2: Deterministic Control Flow (Order-Independent)" << std::endl;
        auto program = createDeterministicBranchTest();
        
        auto verification = interpreter.verifyOrderIndependence(program, 5);
        printVerificationResult(verification, "Deterministic Branching");
        
        if (!verification.results.empty()) {
            printExecutionResult(verification.results[0], "Sample Execution");
        }
    }
    
    // Test 3: Shared Memory (with proper synchronization)
    {
        std::cout << "TEST 3: Shared Memory Access (Order-Independent with Barriers)" << std::endl;
        auto program = createSharedMemoryTest();
        
        auto verification = interpreter.verifyOrderIndependence(program, 3);
        printVerificationResult(verification, "Shared Memory with Barriers");
    }
    
    // Test 4: Order-Dependent Program (should fail)
    {
        std::cout << "TEST 4: Order-Dependent Program (Should Fail Verification)" << std::endl;
        auto program = createOrderDependentTest();
        
        auto verification = interpreter.verifyOrderIndependence(program, 5);
        printVerificationResult(verification, "Order-Dependent Test");
    }
    
    // Demonstrate different thread orderings
    {
        std::cout << "DEMO: Thread Ordering Strategies" << std::endl;
        std::cout << "=================================" << std::endl;
        
        uint32_t threadCount = 8;
        std::vector<ThreadOrdering> orderings = {
            ThreadOrdering::sequential(threadCount),
            ThreadOrdering::reverseSequential(threadCount),
            ThreadOrdering::evenOddInterleaved(threadCount),
            ThreadOrdering::waveInterleaved(threadCount, 4),
            ThreadOrdering::random(threadCount, 123)
        };
        
        for (const auto& ordering : orderings) {
            std::cout << ordering.description << ": ";
            for (size_t i = 0; i < std::min(size_t(8), ordering.executionOrder.size()); ++i) {
                std::cout << ordering.executionOrder[i];
                if (i < std::min(size_t(8), ordering.executionOrder.size()) - 1) {
                    std::cout << " → ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "All tests completed!" << std::endl;
    return 0;
}