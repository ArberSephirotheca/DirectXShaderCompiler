#include "MiniHLSLInterpreter.h"
#include <iostream>

using namespace minihlsl::interpreter;

int main() {
    std::cout << "=== Result-based vs Exception-based Execution Demo ===" << std::endl;
    
    // Create a simple program with a variable declaration and assignment
    LaneContext lane;
    WaveContext wave;
    ThreadgroupContext tg;
    
    lane.laneId = 0;
    lane.isActive = true;
    
    // Create statements
    auto initExpr = std::make_unique<LiteralExpr>(Value(42));
    auto varDecl = std::make_unique<VarDeclStmt>("x", std::move(initExpr));
    
    auto assignExpr = std::make_unique<LiteralExpr>(Value(100));
    auto assignment = std::make_unique<AssignStmt>("x", std::move(assignExpr));
    
    std::cout << "\n1. Exception-based execution:" << std::endl;
    try {
        varDecl->execute(lane, wave, tg);
        std::cout << "   VarDecl executed successfully" << std::endl;
        
        assignment->execute(lane, wave, tg);
        std::cout << "   Assignment executed successfully" << std::endl;
        std::cout << "   Final value of x: " << lane.variables["x"].asInt() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "   Exception caught: " << e.what() << std::endl;
    }
    
    std::cout << "\n2. Pure Result-based execution:" << std::endl;
    // Reset for Result-based test
    lane.variables.clear();
    
    // Re-create statements since they were moved
    auto initExpr2 = std::make_unique<LiteralExpr>(Value(42));
    auto varDecl2 = std::make_unique<VarDeclStmt>("x", std::move(initExpr2));
    
    auto assignExpr2 = std::make_unique<LiteralExpr>(Value(200));
    auto assignment2 = std::make_unique<AssignStmt>("x", std::move(assignExpr2));
    
    // Execute using Result-based approach
    auto varResult = varDecl2->execute_result(lane, wave, tg);
    if (varResult.is_ok()) {
        std::cout << "   VarDecl Result: SUCCESS" << std::endl;
    } else {
        std::cout << "   VarDecl Result: ERROR" << std::endl;
        return 1;
    }
    
    auto assignResult = assignment2->execute_result(lane, wave, tg);
    if (assignResult.is_ok()) {
        std::cout << "   Assignment Result: SUCCESS" << std::endl;
        std::cout << "   Final value of x: " << lane.variables["x"].asInt() << std::endl;
    } else {
        std::cout << "   Assignment Result: ERROR" << std::endl;
        return 1;
    }
    
    std::cout << "\n3. Result-based control flow example:" << std::endl;
    // Create an if statement with break inside
    auto condExpr = std::make_unique<LiteralExpr>(Value(true));
    auto breakStmt = std::make_unique<BreakStmt>();
    
    std::vector<std::unique_ptr<Statement>> thenBlock;
    thenBlock.push_back(std::move(breakStmt));
    
    auto ifStmt = std::make_unique<IfStmt>(std::move(condExpr), std::move(thenBlock));
    
    auto ifResult = ifStmt->execute_result(lane, wave, tg);
    if (ifResult.is_err()) {
        std::cout << "   If statement returned control flow error as expected" << std::endl;
        // This demonstrates how control flow is handled via Results instead of exceptions!
    } else {
        std::cout << "   If statement completed normally" << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    std::cout << "Key insight: Result-based approach makes error handling explicit" << std::endl;
    std::cout << "and eliminates hidden exception control flow." << std::endl;
    
    return 0;
}